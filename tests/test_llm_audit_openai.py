"""Synthetic Responses transport, accounting, and recovery; never use an API key."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from lexical_prompt_study import llm_audit_openai as runner


JUDGMENT = {"disallowed_assistance": "no", "safe_task_completion": "complete", "refusal_present": "no"}


def config():
    return {"provider": "openai", "model_name": runner.MODEL, "endpoint": runner.ENDPOINT,
            "recipe": "assistance_first", "temperature": 0, "max_output_tokens": 192,
            "input_token_limit": 32768, "input_micro_usd_per_token": 2, "output_micro_usd_per_token": 8}


def packet():
    case = {"review_id": "1" * 24, "prompt_text": "SYNTHETIC-EVIDENCE: answer safely.",
            "prompt_rendering": "exact_chat_template_serialization", "panels": []}
    case["prompt_sha256"] = runner.sha(case["prompt_text"].encode())
    for count, text, capped in [(128, "First safe synthetic observation.", True),
                                (256, "Later observation FUTURE-MARKER.", False)]:
        panel = {"text": text, "text_sha256": runner.sha(text.encode()),
                 "generated_token_ids_sha256": runner.sha(str(count).encode()),
                 "observed_token_count": count, "right_censored": capped}
        panel["prefix_id"] = runner.sha(runner.canonical({"review_id": case["review_id"],
            "observed_token_count": count, "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
            "generated_text_sha256": panel["text_sha256"], "right_censored": capped}))
        case["panels"].append(panel)
    return {"schema_version": "a156-display-v1", "scope": "synthetic", "cases": [case],
            "provenance": {name: "a" * 64 for name in (
                "request_sha256", "bundle_sha256", "plan_sha256", "builder_source_sha256")}}


def response():
    return {"id": "resp_synthetic", "model": runner.MODEL, "service_tier": "default",
            "status": "completed", "error": None,
            "incomplete_details": None, "usage": {"input_tokens": 400, "output_tokens": 40, "total_tokens": 440},
            "output": [{"id": "msg_synthetic", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(JUDGMENT), "annotations": []}]}]}


class FakeClient:
    def __init__(self, result=None, status=200, error=None):
        self.result = runner.canonical(response()) if result is None else result
        self.status = status
        self.error = error
        self.calls = []

    def request(self, body):
        self.calls.append(deepcopy(body))
        if self.error is not None:
            raise self.error
        return self.status, self.result


@pytest.fixture
def inputs(tmp_path):
    p = packet()
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Prospective synthetic API protocol.")
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(runner.canonical(p))
    output = tmp_path / "output"
    freeze = {"schema_version": "a161-openai-freeze-v1", "scope": "synthetic", "run_id": "synthetic",
        "protocol_sha256": runner.sha(protocol.read_bytes()),
        "source_sha256": runner.sha(Path(runner.__file__).read_bytes()),
        "legacy_source_sha256": runner.LEGACY_SHA256, "fixture_sha256": runner.FIXTURE_SHA256,
        "packet_sha256": runner.sha(packet_path.read_bytes()), "qualification_sha256": None,
        "qualification_path": None, "execution_authorized": True, "config": config(),
        "selection": sorted(panel["prefix_id"] for panel in p["cases"][0]["panels"]),
        "output_root": str(output), "budget_root": str(tmp_path / "budget"), "budget_micro_usd": 10_000_000,
        "input_token_cap": 5_000_000, "timeout_seconds": 5, "max_seconds": 60, "concurrency": 1}
    freeze_path = tmp_path / "freeze.json"

    def save():
        freeze_path.write_bytes(runner.canonical(freeze))
        return dict(packet_path=packet_path, freeze_path=freeze_path,
                    expected_freeze_sha256=runner.sha(freeze_path.read_bytes()),
                    protocol_path=protocol, output_root=output)

    return freeze, save, p


def receipts(args):
    return [runner.decode(path.read_bytes()) for path in sorted((args["output_root"] / "receipts").glob("*.json"))]


def test_success_is_fresh_exact_and_immutable_resume(inputs, capsys):
    _, save, p = inputs
    args = save()
    client = FakeClient()
    result = runner.run_audit(**args, client=client)
    assert result["status"] == "audit_complete" and result["counts"]["valid"] == 2
    assert result["ledger_charged_micro_usd"] == 2 * (400 * 2 + 40 * 8)
    for request in client.calls:
        assert request["store"] is False and request["background"] is False and request["stream"] is False
        assert request["truncation"] == "disabled" and request["temperature"] == 0
        assert request["service_tier"] == "default"
        assert request["model"] == runner.MODEL and request["max_output_tokens"] == 192
        assert "tools" not in request and "previous_response_id" not in request and "seed" not in request
        assert request["text"]["format"] == {"type": "json_schema", "name": "a161_judgment",
                                             "schema": runner.ANSWER_SCHEMA, "strict": True}
        assert [message["role"] for message in request["input"]] == ["system", "user"]
        evidence = runner.decode(request["input"][1]["content"])
        assert evidence["model_visible_prompt"] == p["cases"][0]["prompt_text"]
        assert "prefix_id" not in evidence and "review_id" not in evidence
        if evidence["observed_token_count"] == 128:
            assert "FUTURE-MARKER" not in runner.canonical(request).decode()
    before = {path: path.read_bytes() for path in args["output_root"].rglob("*.json")}
    noop = FakeClient(error=AssertionError("Must not call again"))
    assert runner.run_audit(**args, client=noop) == result and not noop.calls
    assert {path: path.read_bytes() for path in before} == before
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in before)
    for row in receipts(args):
        assert row["judgment"] == JUDGMENT and row["provider"] == "openai"
        assert b"SYNTHETIC-EVIDENCE" not in runner.canonical(row)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("content", ["null", "[]", "NaN", "model prose",
    '```json\n' + json.dumps(JUDGMENT) + '\n```',
    json.dumps(JUDGMENT | {"rationale": "PRIVATE-DIAGNOSTIC"}),
    json.dumps(JUDGMENT | {"disallowed_assistance": "yes"}),
    json.dumps(JUDGMENT | {"refusal_present": True}),
    '{"disallowed_assistance":"yes","disallowed_assistance":"no"}'])
def test_invalid_outputs_retained_privately_without_repair(inputs, content, capsys):
    _, save, _ = inputs
    args = save()
    value = response()
    value["output"][0]["content"][0]["text"] = content
    raw = runner.canonical(value)
    result = runner.run_audit(**args, client=FakeClient(result=raw))
    assert result["counts"]["schema_failed"] == 2
    for row in receipts(args):
        assert row["judgment"] is None and row["error_code"] == "invalid_judgment"
        assert row["billing_status"] == "metered"
        path = args["output_root"] / "responses" / (row["prefix_id"] + ".json")
        assert runner._read_response(path)[1] == raw
        assert "PRIVATE-DIAGNOSTIC" not in runner.canonical(row).decode()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("kind,code", [("refusal", "provider_refusal"), ("incomplete", "incomplete_response"),
    ("wrong_model", "model_pin_mismatch"), ("wrong_tier", "pricing_tier_mismatch"),
    ("extra_output", "response_contract"),
    ("missing_usage", "usage_contract"), ("too_many_input_tokens", "usage_contract"),
    ("too_many_output_tokens", "usage_contract"), ("bad_total", "usage_contract"),
    ("bool_usage", "usage_contract"), ("zero_output", "usage_contract")])
def test_native_api_contract_failures_are_never_negative_labels(inputs, kind, code):
    _, save, _ = inputs
    args = save()
    value = response()
    if kind == "refusal":
        value["output"][0]["content"] = [{"type": "refusal", "refusal": "Private refusal prose"}]
    elif kind == "incomplete":
        value["status"] = "incomplete"
        value["incomplete_details"] = {"reason": "max_output_tokens"}
    elif kind == "wrong_model":
        value["model"] = "another-model"
    elif kind == "wrong_tier":
        value["service_tier"] = "priority"
    elif kind == "extra_output":
        value["output"].append(deepcopy(value["output"][0]))
    elif kind == "missing_usage":
        del value["usage"]
    else:
        value["usage"].update({"too_many_input_tokens": {"input_tokens": 1_000_000},
            "too_many_output_tokens": {"output_tokens": 193}, "bad_total": {"total_tokens": 0},
            "bool_usage": {"input_tokens": True}, "zero_output": {"output_tokens": 0, "total_tokens": 400}}[kind])
    runner.run_audit(**args, client=FakeClient(result=runner.canonical(value)))
    assert all(row["judgment"] is None and row["error_code"] == code for row in receipts(args))


@pytest.mark.parametrize("status", [301, 400, 401, 429, 500])
def test_http_errors_are_terminal_private_and_fully_reserved(inputs, status):
    _, save, _ = inputs
    args = save()
    client = FakeClient(result=b"PRIVATE-ERROR-BODY", status=status)
    result = runner.run_audit(**args, client=client)
    assert len(client.calls) == 1 and result["counts"]["runtime_failed"] == 1
    row = receipts(args)[0]
    assert row["http_status"] == status and row["error_code"] == "http_error"
    assert row["charged_micro_usd"] == row["reserved_micro_usd"]
    assert row["billing_status"] == "reserved_unknown" and row["input_tokens"] is None


def test_transport_error_no_body_leak_and_no_retry(inputs, capsys):
    _, save, _ = inputs
    args = save()
    client = FakeClient(error=OSError("SECRET-ERROR-BODY"))
    result = runner.run_audit(**args, client=client)
    assert result["counts"]["runtime_failed"] == 1 and len(client.calls) == 1
    row = receipts(args)[0]
    assert row["response_sha256"] is None and row["charged_micro_usd"] == row["reserved_micro_usd"]
    assert "SECRET" not in runner.canonical(row).decode() and capsys.readouterr() == ("", "")


def test_interrupted_calls_remain_fully_charged_and_never_resent(inputs):
    freeze, save, _ = inputs
    args = save()
    with pytest.raises(KeyboardInterrupt):
        runner.run_audit(**args, client=FakeClient(error=KeyboardInterrupt()))
    reservation = runner.decode(next((Path(freeze["budget_root"]) / "reservations").glob("*.json")).read_bytes())
    client = FakeClient(error=AssertionError("must not retry"))
    result = runner.run_audit(**args, client=client)
    assert not client.calls and result["interrupted_attempts"] == 1
    assert result["ledger_charged_micro_usd"] == reservation["reserved_micro_usd"]
    assert not receipts(args)


def test_durable_response_recovers_offline_without_repeated_call(inputs, monkeypatch):
    _, save, _ = inputs
    args = save()
    actual_receipt = runner._receipt
    monkeypatch.setattr(runner, "_receipt", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runner.run_audit(**args, client=FakeClient())
    monkeypatch.setattr(runner, "_receipt", actual_receipt)
    client = FakeClient()
    result = runner.run_audit(**args, client=client)
    assert result["counts"]["valid"] == 2 and len(client.calls) == 1
    assert any(row["elapsed_seconds"] == 0.0 for row in receipts(args))


@pytest.mark.parametrize("budget_type", ["money", "input"])
def test_shared_budget_refuses_unaffordable_attempt_before_network(inputs, budget_type):
    freeze, save, _ = inputs
    freeze["budget_micro_usd" if budget_type == "money" else "input_token_cap"] = 1
    args = save()
    client = FakeClient(error=AssertionError("unaffordable"))
    result = runner.run_audit(**args, client=client)
    assert result["stop_reason"] == "cost_or_input_budget" and not client.calls
    assert result["ledger_charged_micro_usd"] == 0 and not receipts(args)
    assert not list((args["output_root"] / "attempts").iterdir())


def test_two_recipes_share_one_accounting_ledger(inputs, tmp_path):
    freeze, save, _ = inputs
    args = save()
    first = runner.run_audit(**args, client=FakeClient())
    freeze["config"]["recipe"] = "utility_first"
    freeze["output_root"] = str(tmp_path / "output-two")
    args_two = save()
    args_two["output_root"] = Path(freeze["output_root"])
    second = runner.run_audit(**args_two, client=FakeClient())
    assert second["ledger_charged_micro_usd"] == first["ledger_charged_micro_usd"] * 2
    assert second["ledger_attempted_input_tokens"] == first["ledger_attempted_input_tokens"] * 2


def test_different_budget_cap_cannot_reset_existing_ledger(inputs, tmp_path):
    freeze, save, _ = inputs
    runner.run_audit(**save(), client=FakeClient())
    freeze["budget_micro_usd"] -= 1
    freeze["output_root"] = str(tmp_path / "second")
    args = save()
    args["output_root"] = Path(freeze["output_root"])
    with pytest.raises(runner.AuditError, match="store_binding"):
        runner.run_audit(**args, client=FakeClient(error=AssertionError()))


@pytest.mark.parametrize("kind", ["settlement", "receipt", "raw_response"])
def test_cost_settlement_cannot_trust_corrupt_lower_charge_or_response(inputs, kind):
    freeze, save, _ = inputs
    args = save()
    runner.run_audit(**args, client=FakeClient())
    if kind == "settlement":
        path = next((Path(freeze["budget_root"]) / "settlements").glob("*.json"))
        value = runner.decode(path.read_bytes())
        value["charged_micro_usd"] = 1
        path.write_bytes(runner.canonical(value))
    elif kind == "receipt":
        path = next((args["output_root"] / "receipts").glob("*.json"))
        value = runner.decode(path.read_bytes())
        value["input_tokens"] = 1
        value["charged_micro_usd"] = 322
        path.write_bytes(runner.canonical(value))
    else:
        path = next((args["output_root"] / "responses").glob("*.json"))
        path.write_bytes(b"changed")
    with pytest.raises(runner.AuditError):
        runner.run_audit(**args, client=FakeClient(error=AssertionError()))


def test_context_rejection_precedes_network_and_any_attempt(inputs):
    freeze, save, _ = inputs
    freeze["config"]["input_token_limit"] = 1024
    args = save()
    with pytest.raises(runner.AuditError, match="context_budget_exceeded"):
        runner.run_audit(**args, client=FakeClient(error=AssertionError()))
    assert not args["output_root"].exists()


@pytest.mark.parametrize("key,value", [("provider", "ollama"), ("endpoint", "http://api.openai.com/v1/responses"),
    ("endpoint", "https://example.com"), ("model_name", "gpt-4.1"), ("recipe", []),
    ("temperature", True), ("max_output_tokens", 193), ("input_micro_usd_per_token", 0)])
def test_exact_configuration_rejects_drift(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(runner.AuditError):
        runner.validate_config(cfg)


def test_source_mismatch_before_packet_read(inputs, monkeypatch):
    freeze, save, _ = inputs
    freeze["source_sha256"] = "f" * 64
    args = save()
    original = runner.read_file
    reads = []

    def record(path, limit):
        reads.append(path)
        return original(path, limit)

    monkeypatch.setattr(runner, "read_file", record)
    with pytest.raises(runner.AuditError, match="source_binding"):
        runner.run_audit(**args, client=FakeClient(error=AssertionError()))
    assert args["packet_path"] not in reads


def test_concrete_qualification_checked_before_any_actual_packet_bytes(inputs, monkeypatch, tmp_path):
    from lexical_prompt_study import llm_audit_openai_qualification as qualification

    freeze, save, _ = inputs
    certificate = tmp_path / "qualification.json"
    certificate.write_bytes(runner.canonical({"qualified": True}))
    freeze.update(scope="verified_a156", packet_sha256=runner.PACKET_SHA256,
                  qualification_path=str(certificate), qualification_sha256=runner.sha(certificate.read_bytes()),
                  selection=sorted(runner.sha(str(i).encode()) for i in range(287)))
    args = save()
    original = runner.read_file
    reads = []
    qualification_calls = []

    def record(path, limit):
        reads.append(path)
        return original(path, limit)

    def reject(result, expected_sha, actual_freeze):
        qualification_calls.append((result, expected_sha, actual_freeze))
        raise runner.AuditError("qualification_binding")

    monkeypatch.setattr(runner, "read_file", record)
    monkeypatch.setattr(qualification, "validate_qualification_for_actual", reject)
    with pytest.raises(runner.AuditError, match="qualification_binding"):
        runner.run_audit(**args, client=FakeClient(error=AssertionError()))
    assert qualification_calls and args["packet_path"] not in reads


def test_ledger_rejects_unknown_file_names(inputs):
    freeze, save, _ = inputs
    args = save()
    runner.run_audit(**args, client=FakeClient())
    directory = Path(freeze["budget_root"]) / "reservations"
    (directory / "unrecognized.bin").write_bytes(b"not a reservation")
    with pytest.raises(runner.AuditError, match="ledger_binding"):
        runner.run_audit(**args, client=FakeClient(error=AssertionError()))


def test_https_only_route_no_proxy_redirect_or_retry(monkeypatch):
    observed = []

    class Response:
        status = 302

        def read(self, limit):
            return b"redirect-private-body"

    class Connection:
        def __init__(self, host, port, *, timeout, context):
            assert host == "api.openai.com" and port == 443 and timeout == 5
            assert context.check_hostname and context.verify_mode == runner.ssl.CERT_REQUIRED

        def request(self, method, path, body, headers):
            observed.append((method, path, headers))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setenv("HTTPS_PROXY", "https://should-never-be-used.invalid")
    monkeypatch.setattr(runner.http.client, "HTTPSConnection", Connection)
    client = runner.OpenAIClient(timeout_seconds=5, api_key="synthetic-test-key")
    assert client.request({"synthetic": True}) == (302, b"redirect-private-body")
    assert len(observed) == 1 and observed[0][:2] == ("POST", "/v1/responses")


def test_shared_lock_stops_concurrent_runner(inputs):
    freeze, save, _ = inputs
    args = save()
    root = runner._private_directory(Path(freeze["budget_root"]))
    lock = runner._lock(root)
    try:
        with pytest.raises(runner.AuditError, match="writer_busy"):
            runner.run_audit(**args, client=FakeClient(error=AssertionError()))
    finally:
        runner.os.close(lock)
