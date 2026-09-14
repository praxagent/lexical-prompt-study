"""A162 clarified rubric; shared frozen OpenAI transport, receipts, and global budget."""

from __future__ import annotations

import argparse
import http.client
import os
from pathlib import Path
import re
import time
from typing import Any

from . import llm_audit_openai as provider
from . import llm_audit_runner as legacy
from .llm_audit_runner import (
    AuditError, MAX_METADATA, MAX_PACKET, PACKET_SHA256, canonical, decode, digest,
    read_file, require, safe_path, sha, validate_packet, write_new,
)

ENGINE_SHA256 = "e82dec801ef850ee3ac068e55eaea27b53352bc07fb4c4f2ab8f3b70c679cbab"
LEGACY_SHA256 = provider.LEGACY_SHA256
DEVELOPMENT_FIXTURE_SHA256 = provider.FIXTURE_SHA256
BUDGET_ROOT = "/data2/PRAX/lexical-prompt-study-data/runs/a161-openai/budget"
BUDGET_PROTOCOL_SHA256 = "4d9d79e4ddbabae7746d2f2ee0c1980db2a8543e7f26acfd5f04f177bcb12cdc"
CONFIG_FIELDS = provider.CONFIG_FIELDS | {"instrument_id", "rubric_sha256"}
FREEZE_FIELDS = provider.FREEZE_FIELDS | {
    "instrument_id", "stage", "engine_source_sha256", "rubric_path", "rubric_sha256",
    "holdout_fixture_path", "holdout_fixture_sha256", "development_fixture_path",
    "development_fixture_sha256", "budget_protocol_sha256",
}
# Reused provider wire format. Its freeze/source/protocol/config/request hashes identify A162.
BINDING_FIELDS = provider.BINDING_FIELDS
ATTEMPT_FIELDS = provider.ATTEMPT_FIELDS
RECEIPT_FIELDS = provider.RECEIPT_FIELDS
STATUSES = provider.STATUSES
ERROR_CODES = provider.ERROR_CODES
_receipt = provider._receipt


def validate_config(config: Any) -> dict:
    require(type(config) is dict and set(config) == CONFIG_FIELDS, "a162_config_schema")
    require(config["instrument_id"] == "a162" and digest(config["rubric_sha256"]), "a162_instrument")
    provider.validate_config({key: config[key] for key in provider.CONFIG_FIELDS})
    return config


def validate_rubric(rubric: Any) -> dict:
    require(type(rubric) is dict and set(rubric) == {"assistance_first", "utility_first"}, "a162_rubric_schema")
    require(all(type(value) is str and value.strip() and len(value.encode("utf-8")) <= 32768
                for value in rubric.values()), "a162_rubric_text")
    return rubric


def validate_freeze(freeze: Any) -> dict:
    require(type(freeze) is dict and set(freeze) == FREEZE_FIELDS
            and freeze["schema_version"] == "a162-freeze-v1" and freeze["instrument_id"] == "a162"
            and type(freeze["stage"]) is str and freeze["stage"] in {"development", "holdout", "actual"}
            and freeze["execution_authorized"] is True, "a162_freeze_schema")
    require(type(freeze["run_id"]) is str
            and re.fullmatch(r"a162-[a-z0-9][a-z0-9_-]{0,57}", freeze["run_id"]), "a162_run_identity")
    expected_scope = "verified_a156" if freeze["stage"] == "actual" else "synthetic"
    require(freeze["scope"] == expected_scope, "a162_stage_scope")
    for key in ("protocol_sha256", "source_sha256", "engine_source_sha256", "legacy_source_sha256",
                "packet_sha256", "fixture_sha256", "rubric_sha256", "holdout_fixture_sha256",
                "development_fixture_sha256", "budget_protocol_sha256"):
        require(digest(freeze[key]), "a162_freeze_digest")
    require(freeze["engine_source_sha256"] == ENGINE_SHA256
            and freeze["legacy_source_sha256"] == LEGACY_SHA256
            and freeze["development_fixture_sha256"] == DEVELOPMENT_FIXTURE_SHA256
            and freeze["holdout_fixture_sha256"] != DEVELOPMENT_FIXTURE_SHA256,
            "a162_dependency_pins")
    require(freeze["fixture_sha256"] == freeze["development_fixture_sha256" if freeze["stage"] == "development"
                                               else "holdout_fixture_sha256"], "a162_active_fixture")
    validate_config(freeze["config"])
    require(freeze["config"]["rubric_sha256"] == freeze["rubric_sha256"], "a162_config_rubric")
    for key in ("output_root", "budget_root", "rubric_path", "holdout_fixture_path", "development_fixture_path"):
        require(type(freeze[key]) is str and str(safe_path(freeze[key])) == freeze[key], "a162_freeze_path")
    output, budget = Path(freeze["output_root"]), Path(freeze["budget_root"])
    require(output != budget and output not in budget.parents and budget not in output.parents, "a162_freeze_path")
    require(freeze["budget_root"] == BUDGET_ROOT
            and freeze["budget_protocol_sha256"] == BUDGET_PROTOCOL_SHA256
            and type(freeze["budget_micro_usd"]) is int and freeze["budget_micro_usd"] == 10_000_000
            and type(freeze["input_token_cap"]) is int and freeze["input_token_cap"] == 5_000_000,
            "a162_existing_global_budget")
    for key, high in (("timeout_seconds", 600), ("max_seconds", 43200)):
        require(type(freeze[key]) is int and 1 <= freeze[key] <= high, "a162_runtime_bounds")
    require(type(freeze["concurrency"]) is int and freeze["concurrency"] == 1, "a162_runtime_bounds")
    selection = freeze["selection"]
    expected_count = {"development": 12, "holdout": 24, "actual": 287}[freeze["stage"]]
    require(type(selection) is list and len(selection) == expected_count
            and all(digest(prefix) for prefix in selection) and selection == sorted(set(selection)),
            "a162_selection")
    if freeze["stage"] == "development":
        require(freeze["qualification_sha256"] is None and freeze["qualification_path"] is None,
                "a162_development_authority")
    else:
        require(digest(freeze["qualification_sha256"]) and type(freeze["qualification_path"]) is str
                and str(safe_path(freeze["qualification_path"])) == freeze["qualification_path"],
                "a162_qualification_authority")
    if freeze["stage"] == "actual":
        require(freeze["packet_sha256"] == PACKET_SHA256, "a162_actual_packet")
    else:
        require(freeze["packet_sha256"] != PACKET_SHA256, "a162_synthetic_packet")
    return freeze


def make_request(case: dict, panel: dict, config: dict, rubric: dict) -> tuple[dict, int]:
    """Caller supplies rubric from the hash-verified file; exact strings enter the request hash."""
    validate_config(config)
    validate_rubric(rubric)
    evidence = {"model_visible_prompt": case["prompt_text"], "observed_response_prefix": panel["text"],
                "observed_token_count": panel["observed_token_count"], "right_censored": panel["right_censored"]}
    body = {"model": config["model_name"], "input": [
        {"role": "system", "content": rubric[config["recipe"]]},
        {"role": "user", "content": canonical(evidence).decode("utf-8")}],
        "text": {"format": {"type": "json_schema", "name": "a161_judgment",
                            "schema": legacy.ANSWER_SCHEMA, "strict": True}},
        "temperature": 0, "max_output_tokens": 192, "store": False, "background": False,
        "stream": False, "truncation": "disabled", "service_tier": "default"}
    return body, len(canonical(body)) + 1024


def _load_inputs(packet_path: Path, freeze_path: Path, expected_freeze_sha256: str,
                 protocol_path: Path, output_root: Path) -> tuple[dict, dict, str, dict]:
    raw = read_file(freeze_path, MAX_METADATA)
    require(digest(expected_freeze_sha256) and sha(raw) == expected_freeze_sha256, "a162_freeze_binding")
    freeze = validate_freeze(decode(raw))
    require(sha(read_file(Path(__file__), MAX_METADATA)) == freeze["source_sha256"], "a162_source_binding")
    require(sha(read_file(Path(provider.__file__), MAX_METADATA)) == ENGINE_SHA256
            and sha(read_file(Path(legacy.__file__), MAX_METADATA)) == LEGACY_SHA256, "a162_engine_binding")
    require(sha(read_file(protocol_path, MAX_METADATA)) == freeze["protocol_sha256"], "a162_protocol_binding")
    require(str(safe_path(output_root)) == freeze["output_root"], "a162_output_binding")
    rubric_raw = read_file(Path(freeze["rubric_path"]), MAX_METADATA)
    require(sha(rubric_raw) == freeze["rubric_sha256"], "a162_rubric_binding")
    rubric = validate_rubric(decode(rubric_raw))
    for name in ("development_fixture", "holdout_fixture"):
        require(sha(read_file(Path(freeze[name + "_path"]), MAX_METADATA)) == freeze[name + "_sha256"],
                "a162_fixture_binding")
    if freeze["stage"] != "development":
        from . import llm_audit_a162_qualification as qualification
        qualification_raw = read_file(Path(freeze["qualification_path"]), MAX_METADATA)
        require(sha(qualification_raw) == freeze["qualification_sha256"], "a162_qualification_binding")
        gate = (qualification.validate_development_for_holdout if freeze["stage"] == "holdout"
                else qualification.validate_qualification_for_actual)
        gate(decode(qualification_raw), freeze["qualification_sha256"], freeze)
    # No selected packet bytes or provider access before all frozen-instrument gates.
    packet_raw = read_file(packet_path, MAX_PACKET)
    require(sha(packet_raw) == freeze["packet_sha256"], "a162_packet_binding")
    packet = validate_packet(decode(packet_raw))
    require(packet["scope"] == freeze["scope"], "a162_packet_scope")
    panels = {panel["prefix_id"] for case in packet["cases"] for panel in case["panels"]}
    require(set(freeze["selection"]) == panels, "a162_selection_binding")
    for case in packet["cases"]:
        for panel in case["panels"]:
            _, estimated = make_request(case, panel, freeze["config"], rubric)
            require(estimated <= freeze["config"]["input_token_limit"], "context_budget_exceeded")
    return freeze, packet, sha(canonical(freeze["config"])), rubric


def run_audit(*, packet_path: Path, freeze_path: Path, expected_freeze_sha256: str,
              protocol_path: Path, output_root: Path, client: provider.OpenAIClient | None = None) -> dict:
    """A162 loop, sharing immutable provider receipt contract and cumulative A161 budget."""
    freeze, packet, config_sha, rubric = _load_inputs(packet_path, freeze_path, expected_freeze_sha256,
                                                     protocol_path, output_root)
    root = provider._private_directory(output_root)
    budget_root = safe_path(freeze["budget_root"])
    # A162 may use, but may not silently initialize/reset, the original global ledger.
    require((budget_root / "manifest.json").is_file(), "a162_existing_budget_required")
    budget_fd = provider._lock(budget_root)
    try:
        run_fd = provider._lock(root)
        try:
            budget_config = {**freeze, "protocol_sha256": freeze["budget_protocol_sha256"]}
            ledger = provider.BudgetLedger(budget_root, budget_config)
            ledger.totals()
            provider._init_store(root, {"schema_version": "a162-store-v1", "instrument_id": "a162",
                "freeze_sha256": expected_freeze_sha256, "config_sha256": config_sha,
                "selected_prefixes": len(freeze["selection"])}, ("attempts", "responses", "receipts"))
            names = {prefix + ".json" for prefix in freeze["selection"]}
            for name in ("attempts", "responses", "receipts"):
                require({path.name for path in (root / name).iterdir()} <= names, "unexpected_store_member")
            panels = {p["prefix_id"]: (c, p) for c in packet["cases"] for p in c["panels"]}
            counts = {status: 0 for status in sorted(STATUSES)}
            interrupted = 0
            stop_reason = "none"
            started = time.monotonic()
            transport = client
            for prefix in freeze["selection"]:
                case, panel = panels[prefix]
                request, estimated = make_request(case, panel, freeze["config"], rubric)
                bindings = provider._bindings(freeze, expected_freeze_sha256, config_sha,
                                               case, panel, request, estimated)
                key = sha(canonical({"freeze_sha256": expected_freeze_sha256, "prefix_id": prefix}))
                attempt = {"schema_version": "a161-openai-attempt-v1", **bindings,
                           "budget_entry_id": key, "reserved_micro_usd": estimated * 2 + 192 * 8}
                attempt_path = root / "attempts" / (prefix + ".json")
                response_path = root / "responses" / (prefix + ".json")
                receipt_path = root / "receipts" / (prefix + ".json")
                if attempt_path.exists():
                    attempt_raw = read_file(attempt_path, MAX_METADATA)
                    require(decode(attempt_raw) == attempt, "attempt_binding")
                    attempt_sha = sha(attempt_raw)
                    ledger.verify_reservation(attempt)
                    if receipt_path.exists():
                        receipt_raw = read_file(receipt_path, MAX_METADATA)
                        receipt = provider.validate_receipt(decode(receipt_raw), bindings, attempt_sha)
                        require(response_path.exists() == (receipt["response_sha256"] is not None), "response_binding")
                        if response_path.exists():
                            require(provider._read_response(response_path)[2] == receipt["response_sha256"], "response_binding")
                        ledger.settle(attempt, receipt, sha(receipt_raw))
                        counts[receipt["status"]] += 1
                        continue
                    if response_path.exists():
                        receipt = _receipt(attempt, attempt_sha, response_path, 0.0)
                    else:
                        interrupted += 1
                        stop_reason = "interrupted_attempt_requires_recovery"
                        break
                else:
                    require(not response_path.exists() and not receipt_path.exists(), "orphaned_result")
                    if time.monotonic() - started + freeze["timeout_seconds"] > freeze["max_seconds"]:
                        stop_reason = "run_time_budget"
                        break
                    if transport is None:
                        transport = provider.OpenAIClient(timeout_seconds=freeze["timeout_seconds"])
                    if not ledger.reserve(attempt, root):
                        stop_reason = "cost_or_input_budget"
                        break
                    attempt_sha = write_new(attempt_path, attempt)
                    call_started = time.monotonic()
                    try:
                        status, raw = transport.request(request)
                    except (AuditError, OSError, http.client.HTTPException):
                        receipt = _receipt(attempt, attempt_sha, None, time.monotonic() - call_started)
                    else:
                        provider._write_response(response_path, status, raw)
                        receipt = _receipt(attempt, attempt_sha, response_path, time.monotonic() - call_started)
                provider.validate_receipt(receipt, bindings, attempt_sha)
                receipt_sha = write_new(receipt_path, receipt)
                ledger.settle(attempt, receipt, receipt_sha)
                counts[receipt["status"]] += 1
                if receipt["status"] == "runtime_failed":
                    stop_reason = receipt["error_code"]
                    break
            charge, inputs = ledger.totals()
            return {"status": "audit_complete" if sum(counts.values()) == len(freeze["selection"])
                    else "audit_incomplete", "instrument_id": "a162", "provider": "openai",
                    "freeze_sha256": expected_freeze_sha256, "selected_prefixes": len(freeze["selection"]),
                    "counts": counts, "interrupted_attempts": interrupted, "stop_reason": stop_reason,
                    "ledger_charged_micro_usd": charge, "ledger_attempted_input_tokens": inputs}
        finally:
            os.close(run_fd)
    finally:
        os.close(budget_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_audit(packet_path=args.packet, freeze_path=args.freeze,
            expected_freeze_sha256=args.freeze_sha256, protocol_path=args.protocol, output_root=args.output)
        print(canonical(result).decode(), end="")
        return 0 if result["status"] == "audit_complete" and not any(
            result["counts"][key] for key in STATUSES - {"valid"}) else 2
    except AuditError as error:
        print(canonical({"status": str(error), "instrument_id": "a162"}).decode(), end="")
        return 2
    except (Exception, KeyboardInterrupt):
        print('{"status":"internal_error","instrument_id":"a162"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
