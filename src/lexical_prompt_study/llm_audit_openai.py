"""Prospective A161 OpenAI judge: inert import, bounded spend, no automatic retries."""

from __future__ import annotations

import argparse
import fcntl
import http.client
import math
import os
from pathlib import Path
import re
import ssl
import time
from typing import Any

from lexical_prompt_study import llm_audit_runner as legacy
from lexical_prompt_study.llm_audit_runner import (
    ANSWER_SCHEMA, AuditError, MAX_METADATA, MAX_PACKET, PACKET_SHA256, SYSTEM_PROMPTS,
    canonical, decode, digest, read_file, require, safe_path, sha, validate_judgment,
    validate_packet, write_new,
)

MODEL = "gpt-4.1-2025-04-14"
ENDPOINT = "https://api.openai.com/v1/responses"
LEGACY_SHA256 = "5ff1708b287e1d43954bb0266862607a169d28f76b27f412c22b0ec4be8f5a0a"
FIXTURE_SHA256 = "1dd2518ce872902bc1c4e4b20be0e2bfffe92fb8513466c4537adbb51ffcb3a8"
MAX_RESPONSE = 1024**2
CONFIG_FIELDS = {"provider", "model_name", "endpoint", "recipe", "temperature",
                 "max_output_tokens", "input_token_limit", "input_micro_usd_per_token",
                 "output_micro_usd_per_token"}
FREEZE_FIELDS = {"schema_version", "scope", "run_id", "protocol_sha256", "source_sha256",
                 "legacy_source_sha256", "packet_sha256", "fixture_sha256",
                 "qualification_sha256", "qualification_path", "execution_authorized",
                 "config", "selection", "output_root", "budget_root", "budget_micro_usd",
                 "input_token_cap", "timeout_seconds", "max_seconds", "concurrency"}
STATUSES = {"valid", "schema_failed", "runtime_failed"}
ERROR_CODES = {"none", "invalid_judgment", "provider_refusal", "incomplete_response",
               "response_contract", "usage_contract", "model_pin_mismatch", "http_error",
               "runtime_error", "pricing_tier_mismatch"}
BINDING_FIELDS = {"freeze_sha256", "protocol_sha256", "source_sha256", "legacy_source_sha256",
                  "packet_sha256", "fixture_sha256", "config_sha256", "request_sha256",
                  "provider", "model_name", "recipe", "review_id", "prefix_id",
                  "observed_token_count", "right_censored", "prompt_sha256", "text_sha256",
                  "generated_token_ids_sha256", "context_estimated_tokens"}
ATTEMPT_FIELDS = BINDING_FIELDS | {"schema_version", "budget_entry_id", "reserved_micro_usd"}
RECEIPT_FIELDS = ATTEMPT_FIELDS | {"judge_kind", "status", "error_code", "attempt_sha256",
    "response_sha256", "http_status", "judgment", "input_tokens", "output_tokens",
    "charged_micro_usd", "billing_status", "elapsed_seconds"}


def validate_config(config: Any) -> dict:
    require(type(config) is dict and set(config) == CONFIG_FIELDS, "config_schema")
    require(config["provider"] == "openai" and config["model_name"] == MODEL
            and config["endpoint"] == ENDPOINT, "provider_config")
    require(type(config["recipe"]) is str and config["recipe"] in SYSTEM_PROMPTS, "recipe_config")
    for key, expected in (("temperature", 0), ("max_output_tokens", 192),
                          ("input_micro_usd_per_token", 2), ("output_micro_usd_per_token", 8)):
        require(type(config[key]) is int and config[key] == expected, "config_bounds")
    require(type(config["input_token_limit"]) is int
            and 1024 <= config["input_token_limit"] <= 131072, "config_bounds")
    return config


def validate_freeze(freeze: Any) -> dict:
    require(type(freeze) is dict and set(freeze) == FREEZE_FIELDS
            and freeze["schema_version"] == "a161-openai-freeze-v1"
            and type(freeze["scope"]) is str and freeze["scope"] in {"synthetic", "verified_a156"}
            and freeze["execution_authorized"] is True, "freeze_schema")
    require(type(freeze["run_id"]) is str
            and re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", freeze["run_id"]), "run_identity")
    for key in ("protocol_sha256", "source_sha256", "legacy_source_sha256", "packet_sha256",
                "fixture_sha256"):
        require(digest(freeze[key]), "freeze_digest")
    require(freeze["legacy_source_sha256"] == LEGACY_SHA256
            and freeze["fixture_sha256"] == FIXTURE_SHA256, "frozen_dependency")
    validate_config(freeze["config"])
    selection = freeze["selection"]
    require(type(selection) is list and 1 <= len(selection) <= 287
            and all(digest(value) for value in selection)
            and selection == sorted(set(selection)), "selection_schema")
    for key in ("output_root", "budget_root"):
        require(type(freeze[key]) is str and str(safe_path(freeze[key])) == freeze[key], "freeze_path")
    require(not (Path(freeze["output_root"]) == Path(freeze["budget_root"])
                 or Path(freeze["output_root"]) in Path(freeze["budget_root"]).parents
                 or Path(freeze["budget_root"]) in Path(freeze["output_root"]).parents), "freeze_path")
    for key, low, high in (("budget_micro_usd", 1, 10_000_000), ("input_token_cap", 1, 5_000_000),
                           ("timeout_seconds", 1, 600), ("max_seconds", 1, 43200)):
        require(type(freeze[key]) is int and low <= freeze[key] <= high, "freeze_bounds")
    require(type(freeze["concurrency"]) is int and freeze["concurrency"] == 1, "freeze_bounds")
    if freeze["scope"] == "verified_a156":
        require(freeze["packet_sha256"] == PACKET_SHA256 and len(selection) == 287
                and digest(freeze["qualification_sha256"])
                and type(freeze["qualification_path"]) is str
                and str(safe_path(freeze["qualification_path"])) == freeze["qualification_path"],
                "actual_authority")
    else:
        require(freeze["qualification_sha256"] is None and freeze["qualification_path"] is None,
                "synthetic_authority")
    return freeze


def make_request(case: dict, panel: dict, config: dict) -> tuple[dict, int]:
    """Build exactly one fresh request; byte bound includes all input and schema."""
    validate_config(config)
    evidence = {"model_visible_prompt": case["prompt_text"], "observed_response_prefix": panel["text"],
                "observed_token_count": panel["observed_token_count"],
                "right_censored": panel["right_censored"]}
    body = {"model": config["model_name"], "input": [
        {"role": "system", "content": SYSTEM_PROMPTS[config["recipe"]]},
        {"role": "user", "content": canonical(evidence).decode("utf-8")}],
        "text": {"format": {"type": "json_schema", "name": "a161_judgment",
                            "schema": ANSWER_SCHEMA, "strict": True}},
        "temperature": 0, "max_output_tokens": 192, "store": False, "background": False,
        "stream": False, "truncation": "disabled", "service_tier": "default"}
    # Count the whole serialized request, deliberately including non-input keys.
    estimated = len(canonical(body)) + 1024
    return body, estimated


def _load_inputs(packet_path: Path, freeze_path: Path, expected_freeze_sha256: str,
                 protocol_path: Path, output_root: Path) -> tuple[dict, dict, str]:
    raw = read_file(freeze_path, MAX_METADATA)
    require(digest(expected_freeze_sha256) and sha(raw) == expected_freeze_sha256, "freeze_binding")
    freeze = validate_freeze(decode(raw))
    require(sha(read_file(Path(__file__), MAX_METADATA)) == freeze["source_sha256"], "source_binding")
    require(sha(read_file(Path(legacy.__file__), MAX_METADATA)) == LEGACY_SHA256, "dependency_binding")
    require(sha(read_file(protocol_path, MAX_METADATA)) == freeze["protocol_sha256"], "protocol_binding")
    require(str(safe_path(output_root)) == freeze["output_root"], "output_binding")
    if freeze["scope"] == "verified_a156":
        qualification_raw = read_file(Path(freeze["qualification_path"]), MAX_METADATA)
        require(sha(qualification_raw) == freeze["qualification_sha256"], "qualification_binding")
        # The concrete synthetic certificate is checked before any actual evidence is read.
        from lexical_prompt_study.llm_audit_openai_qualification import validate_qualification_for_actual
        validate_qualification_for_actual(decode(qualification_raw),
                                         freeze["qualification_sha256"], freeze)
    packet_raw = read_file(packet_path, MAX_PACKET)
    require(sha(packet_raw) == freeze["packet_sha256"], "packet_binding")
    packet = validate_packet(decode(packet_raw))
    require(packet["scope"] == freeze["scope"], "packet_scope")
    panels = {panel["prefix_id"] for case in packet["cases"] for panel in case["panels"]}
    require(set(freeze["selection"]) <= panels, "selection_binding")
    if freeze["scope"] == "verified_a156":
        require(set(freeze["selection"]) == panels, "actual_selection_coverage")
    # Refuse the entire run before network if any selected input exceeds its bound.
    for case in packet["cases"]:
        for panel in case["panels"]:
            if panel["prefix_id"] in freeze["selection"]:
                _, estimated = make_request(case, panel, freeze["config"])
                require(estimated <= freeze["config"]["input_token_limit"], "context_budget_exceeded")
    return freeze, packet, sha(canonical(freeze["config"]))


class OpenAIClient:
    """Verified HTTPS to one fixed route; http.client does not use proxy env vars."""

    def __init__(self, *, timeout_seconds: int, api_key: str | None = None):
        self.timeout_seconds = timeout_seconds
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        require(type(self._api_key) is str and bool(self._api_key.strip())
                and "\n" not in self._api_key and "\r" not in self._api_key, "credential_unavailable")

    def request(self, body: dict) -> tuple[int, bytes]:
        connection = http.client.HTTPSConnection("api.openai.com", 443,
            timeout=self.timeout_seconds, context=ssl.create_default_context())
        try:
            connection.request("POST", "/v1/responses", body=canonical(body), headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + self._api_key})
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE + 1)
            require(len(raw) <= MAX_RESPONSE, "runtime_error")
            # Return errors/redirects for private retention, never follow them.
            return response.status, raw
        except (OSError, http.client.HTTPException, AuditError):
            raise AuditError("runtime_error") from None
        finally:
            connection.close()


def _write_response(path: Path, status: int, raw: bytes) -> str:
    envelope = canonical({"schema_version": "a161-openai-response-v1", "http_status": status,
                          "body_hex": raw.hex()})
    require(type(status) is int and 100 <= status <= 599 and type(raw) is bytes
            and len(raw) <= MAX_RESPONSE, "response_contract")
    path = safe_path(path)
    pending = path.with_name(path.name + ".pending")
    try:
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(envelope)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(pending, path, follow_symlinks=False)
        pending.unlink()
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        raise AuditError("immutable_write_failed") from None
    return sha(envelope)


def _read_response(path: Path) -> tuple[int, bytes, str]:
    raw = read_file(path, 2 * MAX_RESPONSE + MAX_METADATA)
    envelope = decode(raw)
    require(type(envelope) is dict and set(envelope) == {"schema_version", "http_status", "body_hex"}
            and envelope["schema_version"] == "a161-openai-response-v1"
            and type(envelope["http_status"]) is int and 100 <= envelope["http_status"] <= 599
            and type(envelope["body_hex"]) is str, "response_binding")
    try:
        body = bytes.fromhex(envelope["body_hex"])
    except ValueError:
        raise AuditError("response_binding") from None
    require(len(body) <= MAX_RESPONSE and body.hex() == envelope["body_hex"], "response_binding")
    return envelope["http_status"], body, sha(raw)


def response_usage(value: Any, estimated: int) -> tuple[int, int]:
    require(type(value) is dict and type(value.get("usage")) is dict, "usage_contract")
    usage = value["usage"]
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    require(type(input_tokens) is int and 0 < input_tokens <= estimated
            and type(output_tokens) is int and 0 <= output_tokens <= 192
            and type(usage.get("total_tokens")) is int
            and usage["total_tokens"] == input_tokens + output_tokens, "usage_contract")
    return input_tokens, output_tokens


def response_judgment(value: Any) -> dict:
    require(type(value) is dict and value.get("model") == MODEL, "model_pin_mismatch")
    require(value.get("status") == "completed" and value.get("error") is None
            and value.get("incomplete_details") is None, "incomplete_response")
    output = value.get("output")
    require(type(output) is list and len(output) == 1, "response_contract")
    message = output[0]
    require(type(message) is dict and message.get("type") == "message"
            and message.get("role") == "assistant" and message.get("status") == "completed"
            and type(message.get("content")) is list and len(message["content"]) == 1,
            "response_contract")
    content = message["content"][0]
    require(type(content) is dict, "response_contract")
    require(content.get("type") != "refusal", "provider_refusal")
    require(content.get("type") == "output_text" and type(content.get("text")) is str
            and content.get("annotations", []) == [], "response_contract")
    try:
        return validate_judgment(decode(content["text"]))
    except AuditError:
        raise AuditError("invalid_judgment") from None


def validate_receipt(receipt: Any, bindings: dict, attempt_sha256: str) -> dict:
    require(type(bindings) is dict and set(bindings) == BINDING_FIELDS, "receipt_binding")
    require(type(receipt) is dict and set(receipt) == RECEIPT_FIELDS
            and receipt["schema_version"] == "a161-openai-receipt-v1"
            and receipt["judge_kind"] == "llm" and receipt["provider"] == "openai"
            and all(receipt[key] == value for key, value in bindings.items())
            and receipt["attempt_sha256"] == attempt_sha256
            and type(receipt["status"]) is str and receipt["status"] in STATUSES
            and type(receipt["error_code"]) is str and receipt["error_code"] in ERROR_CODES,
            "receipt_binding")
    require(digest(receipt["budget_entry_id"])
            and receipt["budget_entry_id"] == sha(canonical({"freeze_sha256": bindings["freeze_sha256"],
                                                          "prefix_id": bindings["prefix_id"]}))
            and type(receipt["reserved_micro_usd"]) is int
            and receipt["reserved_micro_usd"] == bindings["context_estimated_tokens"] * 2 + 192 * 8,
            "receipt_cost")
    require(type(receipt["elapsed_seconds"]) in {int, float}
            and math.isfinite(receipt["elapsed_seconds"]) and receipt["elapsed_seconds"] >= 0,
            "receipt_timing")
    require((receipt["response_sha256"] is None and receipt["http_status"] is None)
            or (digest(receipt["response_sha256"]) and type(receipt["http_status"]) is int
                and 100 <= receipt["http_status"] <= 599), "receipt_response")
    require(type(receipt["charged_micro_usd"]) is int
            and 0 < receipt["charged_micro_usd"] <= receipt["reserved_micro_usd"], "receipt_cost")
    if receipt["billing_status"] == "metered":
        require(digest(receipt["response_sha256"])
                and type(receipt["input_tokens"]) is int
                and 0 < receipt["input_tokens"] <= bindings["context_estimated_tokens"]
                and type(receipt["output_tokens"]) is int and 0 <= receipt["output_tokens"] <= 192
                and receipt["charged_micro_usd"] == receipt["input_tokens"] * 2
                + receipt["output_tokens"] * 8, "receipt_cost")
    else:
        require(receipt["billing_status"] == "reserved_unknown" and receipt["input_tokens"] is None
                and receipt["output_tokens"] is None
                and receipt["charged_micro_usd"] == receipt["reserved_micro_usd"], "receipt_cost")
    if receipt["status"] == "valid":
        validate_judgment(receipt["judgment"])
        require(receipt["error_code"] == "none" and receipt["http_status"] == 200
                and receipt["billing_status"] == "metered" and receipt["output_tokens"] > 0,
                "receipt_result")
    else:
        allowed = {"schema_failed": {"invalid_judgment", "provider_refusal", "incomplete_response",
                                      "response_contract", "usage_contract"},
                   "runtime_failed": {"http_error", "runtime_error", "model_pin_mismatch",
                                      "pricing_tier_mismatch"}}
        require(receipt["judgment"] is None and receipt["error_code"] in allowed[receipt["status"]],
                "receipt_failure")
    return receipt


def _private_directory(path: Path) -> Path:
    path = safe_path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(path.is_dir() and path.stat().st_mode & 0o077 == 0, "output_permissions")
    return path


def _lock(root: Path) -> int:
    fd = os.open(root / ".writer.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise AuditError("writer_busy") from None
    return fd


def _init_store(root: Path, header: dict, children: tuple[str, ...]) -> None:
    require(not list(root.rglob("*.pending")), "pending_write_requires_recovery")
    path = root / "manifest.json"
    if path.exists():
        require(decode(read_file(path, MAX_METADATA)) == header, "store_binding")
    else:
        require({item.name for item in root.iterdir()} == {".writer.lock"}, "store_not_empty")
        write_new(path, header)
    for name in children:
        _private_directory(root / name)
    require({item.name for item in root.iterdir()} == {"manifest.json", ".writer.lock", *children},
            "unexpected_store_member")


class BudgetLedger:
    """Caller holds a shared-root exclusive lock for reservation through response storage."""

    def __init__(self, root: Path, freeze: dict):
        self.root = root
        self.cap = freeze["budget_micro_usd"]
        self.input_cap = freeze["input_token_cap"]
        _init_store(root, {"schema_version": "a161-openai-budget-v1", "provider": "openai",
            "model_name": MODEL, "protocol_sha256": freeze["protocol_sha256"],
            "budget_micro_usd": self.cap, "input_token_cap": self.input_cap,
            "input_micro_usd_per_token": 2, "output_micro_usd_per_token": 8}, ("reservations", "settlements"))

    def totals(self) -> tuple[int, int]:
        charge, inputs = 0, 0
        for directory in ("reservations", "settlements"):
            require(all(digest(path.stem) and path.name == path.stem + ".json"
                        for path in (self.root / directory).iterdir()), "ledger_binding")
        reservations = {path.stem: path for path in (self.root / "reservations").iterdir()}
        settlements = {path.stem: path for path in (self.root / "settlements").iterdir()}
        require(set(settlements) <= set(reservations), "ledger_binding")
        for key, path in reservations.items():
            require(digest(key) and path.name == key + ".json", "ledger_binding")
            row = decode(read_file(path, MAX_METADATA))
            require(type(row) is dict and set(row) == {"schema_version", "budget_entry_id", "output_root",
                "freeze_sha256", "prefix_id", "attempt_sha256", "reserved_micro_usd", "input_tokens"}
                and row["schema_version"] == "a161-openai-reservation-v1"
                and row["budget_entry_id"] == key and digest(row["freeze_sha256"])
                and digest(row["prefix_id"]) and digest(row["attempt_sha256"])
                and key == sha(canonical({"freeze_sha256": row["freeze_sha256"], "prefix_id": row["prefix_id"]}))
                and type(row["input_tokens"]) is int and row["input_tokens"] > 0
                and type(row["reserved_micro_usd"]) is int
                and row["reserved_micro_usd"] == row["input_tokens"] * 2 + 192 * 8, "ledger_binding")
            require(type(row["output_root"]) is str
                    and str(safe_path(row["output_root"])) == row["output_root"], "ledger_binding")
            amount = row["reserved_micro_usd"]
            inputs += row["input_tokens"]  # A permanent attempted-input bound, even after settlement.
            if key in settlements:
                settled = decode(read_file(settlements[key], MAX_METADATA))
                require(settlements[key].name == key + ".json" and type(settled) is dict
                        and set(settled) == {"schema_version", "budget_entry_id", "reservation_sha256",
                                            "receipt_sha256", "charged_micro_usd"}
                        and settled["schema_version"] == "a161-openai-settlement-v1"
                        and settled["budget_entry_id"] == key
                        and settled["reservation_sha256"] == sha(read_file(path, MAX_METADATA))
                        and digest(settled["receipt_sha256"])
                        and type(settled["charged_micro_usd"]) is int
                        and 0 < settled["charged_micro_usd"] <= amount, "ledger_binding")
                # A settlement cannot lower the ceiling using a bare claimed receipt hash.
                run_root = Path(row["output_root"])
                attempt_raw = read_file(run_root / "attempts" / (row["prefix_id"] + ".json"), MAX_METADATA)
                attempt = decode(attempt_raw)
                require(type(attempt) is dict and set(attempt) == ATTEMPT_FIELDS
                        and sha(attempt_raw) == row["attempt_sha256"], "ledger_binding")
                receipt_raw = read_file(run_root / "receipts" / (row["prefix_id"] + ".json"), MAX_METADATA)
                receipt = validate_receipt(decode(receipt_raw),
                    {name: attempt[name] for name in BINDING_FIELDS}, row["attempt_sha256"])
                require(sha(receipt_raw) == settled["receipt_sha256"]
                        and receipt["charged_micro_usd"] == settled["charged_micro_usd"], "ledger_binding")
                if receipt["response_sha256"] is not None:
                    require(sha(read_file(run_root / "responses" / (row["prefix_id"] + ".json"),
                                          2 * MAX_RESPONSE + MAX_METADATA)) == receipt["response_sha256"],
                            "ledger_binding")
                amount = settled["charged_micro_usd"]
            charge += amount
        require(charge <= self.cap and inputs <= self.input_cap, "ledger_cap_exceeded")
        return charge, inputs

    def reserve(self, attempt: dict, output_root: Path) -> bool:
        key = attempt["budget_entry_id"]
        path = self.root / "reservations" / (key + ".json")
        require(not path.exists(), "interrupted_attempt_requires_recovery")
        charge, inputs = self.totals()
        if (charge + attempt["reserved_micro_usd"] > self.cap
                or inputs + attempt["context_estimated_tokens"] > self.input_cap):
            return False
        write_new(path, {"schema_version": "a161-openai-reservation-v1", "budget_entry_id": key,
            "output_root": str(safe_path(output_root)),
            "freeze_sha256": attempt["freeze_sha256"], "prefix_id": attempt["prefix_id"],
            "attempt_sha256": sha(canonical(attempt)), "reserved_micro_usd": attempt["reserved_micro_usd"],
            "input_tokens": attempt["context_estimated_tokens"]})
        return True

    def verify_reservation(self, attempt: dict) -> str:
        raw = read_file(self.root / "reservations" / (attempt["budget_entry_id"] + ".json"), MAX_METADATA)
        row = decode(raw)
        require(row.get("attempt_sha256") == sha(canonical(attempt)), "ledger_binding")
        self.totals()
        return sha(raw)

    def settle(self, attempt: dict, receipt: dict, receipt_sha: str) -> None:
        reservation_sha = self.verify_reservation(attempt)
        row = {"schema_version": "a161-openai-settlement-v1", "budget_entry_id": attempt["budget_entry_id"],
            "reservation_sha256": reservation_sha, "receipt_sha256": receipt_sha,
            "charged_micro_usd": receipt["charged_micro_usd"]}
        path = self.root / "settlements" / (attempt["budget_entry_id"] + ".json")
        if path.exists():
            require(decode(read_file(path, MAX_METADATA)) == row, "ledger_binding")
        else:
            write_new(path, row)


def _bindings(freeze: dict, freeze_sha: str, config_sha: str, case: dict, panel: dict,
              request: dict, estimated: int) -> dict:
    return {"freeze_sha256": freeze_sha,
        **{key: freeze[key] for key in ("protocol_sha256", "source_sha256", "legacy_source_sha256",
                                       "packet_sha256", "fixture_sha256")},
        "config_sha256": config_sha, "request_sha256": sha(canonical(request)), "provider": "openai",
        "model_name": MODEL, "recipe": freeze["config"]["recipe"], "review_id": case["review_id"],
        "prefix_id": panel["prefix_id"], "observed_token_count": panel["observed_token_count"],
        "right_censored": panel["right_censored"], "prompt_sha256": case["prompt_sha256"],
        "text_sha256": panel["text_sha256"], "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
        "context_estimated_tokens": estimated}


def _receipt(attempt: dict, attempt_sha: str, response_path: Path | None, elapsed: float) -> dict:
    receipt = {**attempt, "schema_version": "a161-openai-receipt-v1", "judge_kind": "llm",
        "attempt_sha256": attempt_sha, "status": "runtime_failed", "error_code": "runtime_error",
        "judgment": None, "response_sha256": None, "http_status": None, "input_tokens": None,
        "output_tokens": None, "charged_micro_usd": attempt["reserved_micro_usd"],
        "billing_status": "reserved_unknown", "elapsed_seconds": elapsed}
    if response_path is None:
        return receipt
    status, raw, response_sha = _read_response(response_path)
    receipt.update(response_sha256=response_sha, http_status=status)
    if status != 200:
        receipt["error_code"] = "http_error"
        return receipt
    try:
        value = decode(raw)
    except AuditError:
        receipt.update(status="schema_failed", error_code="response_contract")
        return receipt
    if type(value) is not dict or value.get("model") != MODEL:
        receipt.update(status="runtime_failed", error_code="model_pin_mismatch")
        return receipt
    if value.get("service_tier") != "default":
        receipt.update(status="runtime_failed", error_code="pricing_tier_mismatch")
        return receipt
    try:
        inp, out = response_usage(value, attempt["context_estimated_tokens"])
        receipt.update(input_tokens=inp, output_tokens=out, charged_micro_usd=inp * 2 + out * 8,
                       billing_status="metered")
    except AuditError:
        receipt.update(status="schema_failed", error_code="usage_contract")
        return receipt
    try:
        judgment = response_judgment(value)
        require(out > 0, "usage_contract")
        receipt.update(status="valid", error_code="none", judgment=judgment)
    except AuditError as error:
        code = str(error)
        receipt.update(status="runtime_failed" if code == "model_pin_mismatch" else "schema_failed",
                       error_code=code)
    return receipt


def run_audit(*, packet_path: Path, freeze_path: Path, expected_freeze_sha256: str,
              protocol_path: Path, output_root: Path, client: OpenAIClient | None = None) -> dict:
    freeze, packet, config_sha = _load_inputs(packet_path, freeze_path, expected_freeze_sha256,
                                             protocol_path, output_root)
    root = _private_directory(output_root)
    budget_root = _private_directory(Path(freeze["budget_root"]))
    budget_fd = _lock(budget_root)
    try:
        run_fd = _lock(root)
        try:
            ledger = BudgetLedger(budget_root, freeze)
            ledger.totals()
            _init_store(root, {"schema_version": "a161-openai-store-v1",
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
                request, estimated = make_request(case, panel, freeze["config"])
                bindings = _bindings(freeze, expected_freeze_sha256, config_sha, case, panel, request, estimated)
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
                        receipt = validate_receipt(decode(receipt_raw), bindings, attempt_sha)
                        require(response_path.exists() == (receipt["response_sha256"] is not None), "response_binding")
                        if response_path.exists():
                            require(_read_response(response_path)[2] == receipt["response_sha256"], "response_binding")
                        ledger.settle(attempt, receipt, sha(receipt_raw))
                        counts[receipt["status"]] += 1
                        continue
                    if response_path.exists():
                        # A durable response can be classified offline after interruption; never repeat the call.
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
                        transport = OpenAIClient(timeout_seconds=freeze["timeout_seconds"])
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
                        _write_response(response_path, status, raw)
                        receipt = _receipt(attempt, attempt_sha, response_path, time.monotonic() - call_started)
                validate_receipt(receipt, bindings, attempt_sha)
                receipt_sha = write_new(receipt_path, receipt)
                ledger.settle(attempt, receipt, receipt_sha)
                counts[receipt["status"]] += 1
                if receipt["status"] == "runtime_failed":
                    stop_reason = receipt["error_code"]
                    break
            charge, inputs = ledger.totals()
            return {"status": "audit_complete" if sum(counts.values()) == len(freeze["selection"])
                    else "audit_incomplete", "provider": "openai", "freeze_sha256": expected_freeze_sha256,
                    "selected_prefixes": len(freeze["selection"]), "counts": counts,
                    "interrupted_attempts": interrupted, "stop_reason": stop_reason,
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
        print(canonical({"status": str(error)}).decode(), end="")
        return 2
    except (Exception, KeyboardInterrupt):
        print('{"status":"internal_error"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
