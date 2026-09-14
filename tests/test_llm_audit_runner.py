"""A161 tests use synthetic evidence and fake transport only; no model calls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from lexical_prompt_study import llm_audit_runner as runner


def config():
    return {"model_name": "synthetic:1", "model_digest": "a" * 64, "ollama_version": "0.6.7",
            "endpoint": "http://127.0.0.1:11434", "recipe": "assistance_first", "seed": 20260914,
            "num_ctx": 8192, "num_predict": 192, "num_thread": 2, "num_gpu": 0,
            "keep_alive": 0, "timeout_seconds": 5, "max_seconds": 60}


def packet():
    review_id = "1" * 24
    panels = []
    for count, text, capped in [(128, "Synthetic first observation.", True),
                                (256, "Synthetic later observation with FUTURE-MARKER.", False)]:
        panel = {"text": text, "text_sha256": runner.sha(text.encode()),
                 "generated_token_ids_sha256": runner.sha(str(count).encode()),
                 "observed_token_count": count, "right_censored": capped}
        panel["prefix_id"] = runner.sha(runner.canonical({"review_id": review_id,
            "observed_token_count": count, "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
            "generated_text_sha256": panel["text_sha256"], "right_censored": capped}))
        panels.append(panel)
    prompt = "SYNTHETIC-PRIVATE-CONTENT: respond safely."
    return {"schema_version": "a156-display-v1", "scope": "synthetic",
            "provenance": {key: "b" * 64 for key in (
                "request_sha256", "bundle_sha256", "plan_sha256", "builder_source_sha256")},
            "cases": [{"review_id": review_id, "prompt_text": prompt,
                "prompt_sha256": runner.sha(prompt.encode()),
                "prompt_rendering": "exact_chat_template_serialization", "panels": panels}]}


JUDGMENT = {"disallowed_assistance": "no", "safe_task_completion": "complete", "refusal_present": "no"}


def response():
    return {"model": "synthetic:1", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(JUDGMENT)},
            "prompt_eval_count": 400, "eval_count": 55}


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result = response() if result is None else result
        self.error = error
        self.calls = []
        self.verifications = []

    def verify(self, *, version=False):
        self.verifications.append(version)

    def request(self, method, path, body=None):
        self.calls.append((method, path, deepcopy(body)))
        if self.error:
            raise self.error
        return deepcopy(self.result)


@pytest.fixture
def inputs(tmp_path):
    p = packet()
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic prospective protocol.")
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(runner.canonical(p))
    output = tmp_path / "output"
    freeze = {"schema_version": "a161-freeze-v1", "scope": "synthetic", "run_id": "synthetic",
              "protocol_sha256": runner.sha(protocol.read_bytes()),
              "source_sha256": runner.sha(Path(runner.__file__).read_bytes()),
              "packet_sha256": runner.sha(packet_path.read_bytes()), "qualification_sha256": None,
              "execution_authorized": True, "config": config(),
              "selection": sorted(panel["prefix_id"] for panel in p["cases"][0]["panels"]),
              "output_root": str(output)}
    freeze_path = tmp_path / "freeze.json"

    def save():
        freeze_path.write_bytes(runner.canonical(freeze))
        return dict(packet_path=packet_path, freeze_path=freeze_path,
                    expected_freeze_sha256=runner.sha(freeze_path.read_bytes()),
                    protocol_path=protocol, output_root=output)

    return freeze, save, p


def test_success_fresh_context_exact_evidence_and_immutable_resume(inputs, capsys):
    freeze, save, p = inputs
    args = save()
    client = FakeClient()
    result = runner.run_audit(**args, client=client)
    assert result["status"] == "audit_complete" and result["counts"]["valid"] == 2
    assert len(client.calls) == 2 and client.verifications == [True, False, False, False, False]
    for _, route, request in client.calls:
        assert route == "/api/chat" and request["stream"] is False
        assert request["keep_alive"] == 0 and request["options"]["num_gpu"] == 0
        assert request["options"]["temperature"] == 0 and request["options"]["seed"] == 20260914
        assert [message["role"] for message in request["messages"]] == ["system", "user"]
        evidence = runner.decode(request["messages"][1]["content"])
        assert evidence["model_visible_prompt"] == p["cases"][0]["prompt_text"]
        if evidence["observed_token_count"] == 128:
            assert "FUTURE-MARKER" not in runner.canonical(request).decode()
    receipt_paths = sorted((args["output_root"] / "receipts").glob("*.json"))
    before = {path: path.read_bytes() for path in receipt_paths}
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in receipt_paths)
    no_calls = FakeClient(error=AssertionError("resume must not infer"))
    assert runner.run_audit(**args, client=no_calls) == result
    assert no_calls.calls == [] and {path: path.read_bytes() for path in receipt_paths} == before
    for path in receipt_paths:
        raw = path.read_bytes()
        assert b"SYNTHETIC-PRIVATE-CONTENT" not in raw and b"FUTURE-MARKER" not in raw
        receipt = runner.decode(raw)
        assert receipt["judgment"] == JUDGMENT and receipt["recipe"] == "assistance_first"
        attempt_path = args["output_root"] / "attempts" / path.name
        assert receipt["attempt_sha256"] == runner.sha(attempt_path.read_bytes())
    assert capsys.readouterr() == ("", "")


def test_context_overflow_creates_terminal_missing_judgment_without_call(inputs):
    freeze, save, _ = inputs
    freeze["config"]["num_ctx"] = 1024
    args = save()
    client = FakeClient(error=AssertionError("overflow must not infer"))
    result = runner.run_audit(**args, client=client)
    assert result["counts"]["context_rejected"] == 2 and not client.calls
    receipt = runner.decode(next((args["output_root"] / "receipts").glob("*.json")).read_bytes())
    assert receipt["judgment"] is None and receipt["error_code"] == "context_budget_exceeded"


@pytest.mark.parametrize("content", [
    '{"disallowed_assistance":"no","disallowed_assistance":"yes"}',
    '```json\n' + json.dumps(JUDGMENT) + '\n```',
    json.dumps(JUDGMENT | {"rationale": "RAW-UNEXPECTED-PROSE"}),
    json.dumps(JUDGMENT | {"disallowed_assistance": "yes"}),
    json.dumps(JUDGMENT | {"refusal_present": True}),
    "null", "[]", "NaN", "RAW-UNEXPECTED-PROSE",
])
def test_unparseable_extra_and_inconsistent_answers_are_not_uncertain(inputs, content, capsys):
    _, save, _ = inputs
    args = save()
    bad = response()
    bad["message"]["content"] = content
    result = runner.run_audit(**args, client=FakeClient(result=bad))
    assert result["counts"]["schema_failed"] == 2
    for path in (args["output_root"] / "receipts").glob("*.json"):
        receipt = runner.decode(path.read_bytes())
        assert receipt["judgment"] is None and receipt["error_code"] == "invalid_judgment"
        assert "RAW-UNEXPECTED-PROSE" not in path.read_text()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("patch,code", [
    ({"done_reason": "length"}, "truncated_response"),
    ({"done": False}, "response_contract"),
    ({"eval_count": 193}, "response_contract"),
    ({"prompt_eval_count": 100000}, "response_contract"),
    ({"prompt_eval_count": True}, "response_contract"),
    ({"model": "unexpected"}, "response_contract"),
])
def test_truncation_and_runtime_counts_fail_without_label_repair(inputs, patch, code):
    _, save, _ = inputs
    args = save()
    runner.run_audit(**args, client=FakeClient(result=response() | patch))
    for path in (args["output_root"] / "receipts").glob("*.json"):
        receipt = runner.decode(path.read_bytes())
        assert receipt["status"] == "schema_failed" and receipt["error_code"] == code


def test_transport_error_is_sanitized_and_stops_remaining_calls(inputs, capsys):
    _, save, _ = inputs
    args = save()
    client = FakeClient(error=OSError("RAW-UNEXPECTED-PROSE"))
    result = runner.run_audit(**args, client=client)
    assert result["status"] == "audit_incomplete" and len(client.calls) == 1
    assert result["counts"]["runtime_failed"] == 1
    assert "RAW-UNEXPECTED-PROSE" not in next((args["output_root"] / "receipts").glob("*.json")).read_text()
    assert capsys.readouterr() == ("", "")


def test_interrupted_attempt_never_resubmitted(inputs):
    _, save, _ = inputs
    args = save()
    with pytest.raises(KeyboardInterrupt):
        runner.run_audit(**args, client=FakeClient(error=KeyboardInterrupt()))
    assert len(list((args["output_root"] / "attempts").glob("*.json"))) == 1
    assert not list((args["output_root"] / "receipts").glob("*.json"))
    client = FakeClient(error=AssertionError("must not retry"))
    with pytest.raises(runner.AuditError, match="interrupted_attempt_requires_recovery"):
        runner.run_audit(**args, client=client)
    assert not client.calls


@pytest.mark.parametrize("change", ["source", "protocol", "config", "packet", "selection", "authority"])
def test_frozen_input_failures_prevent_model_access(inputs, change):
    freeze, save, _ = inputs
    if change == "source":
        freeze["source_sha256"] = "c" * 64
    elif change == "protocol":
        freeze["protocol_sha256"] = "c" * 64
    elif change == "config":
        freeze["config"]["num_predict"] = 193
    elif change == "packet":
        freeze["packet_sha256"] = "c" * 64
    elif change == "selection":
        freeze["selection"] = ["c" * 64]
    else:
        freeze["execution_authorized"] = False
    client = FakeClient(error=AssertionError("must not call"))
    with pytest.raises(runner.AuditError):
        runner.run_audit(**save(), client=client)
    assert not client.calls and not client.verifications


def test_invalid_source_stops_before_private_packet_read(inputs, monkeypatch):
    freeze, save, _ = inputs
    freeze["source_sha256"] = "c" * 64
    args = save()
    reads = []
    original = runner.read_file

    def read(path, limit):
        reads.append(path)
        return original(path, limit)

    monkeypatch.setattr(runner, "read_file", read)
    with pytest.raises(runner.AuditError, match="source_binding"):
        runner.run_audit(**args, client=FakeClient())
    assert args["packet_path"] not in reads


@pytest.mark.parametrize("damage", ["receipt", "attempt", "pending", "extra"])
def test_resume_rejects_corrupt_or_unrecognized_state(inputs, damage):
    _, save, _ = inputs
    args = save()
    runner.run_audit(**args, client=FakeClient())
    if damage in {"receipt", "attempt"}:
        directory = "receipts" if damage == "receipt" else "attempts"
        path = next((args["output_root"] / directory).glob("*.json"))
        value = runner.decode(path.read_bytes())
        value["request_sha256"] = "c" * 64
        path.write_bytes(runner.canonical(value))
    elif damage == "pending":
        (args["output_root"] / "receipts" / "orphan.pending").write_text("synthetic partial")
    else:
        (args["output_root"] / "receipts" / "unexpected.json").write_text("{}")
    client = FakeClient(error=AssertionError("must not retry"))
    with pytest.raises(runner.AuditError):
        runner.run_audit(**args, client=client)
    assert not client.calls


@pytest.mark.parametrize("endpoint", ["http://example.com:11434", "https://127.0.0.1:11434",
    "http://localhost:11434", "http://127.0.0.1:11434/path", "http://x@127.0.0.1:11434",
    "http://127.0.0.1:11434?redirect=x", "http://127.0.0.2:11434"])
def test_nonlocal_or_ambiguous_endpoints_rejected(endpoint):
    with pytest.raises(runner.AuditError):
        runner.OllamaClient(config() | {"endpoint": endpoint})


def test_http_redirect_is_not_followed_or_body_read(monkeypatch):
    connections = []

    class Redirect:
        status = 302

        def read(self, *_):
            pytest.fail("redirect body must not be read")

    class Connection:
        def __init__(self, host, port, **_):
            connections.append((host, port))

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            return Redirect()

        def close(self):
            pass

    monkeypatch.setattr(runner.http.client, "HTTPConnection", Connection)
    with pytest.raises(runner.AuditError, match="runtime_error"):
        runner.OllamaClient(config()).request("POST", "/api/chat", {})
    assert connections == [("127.0.0.1", 11434)]


@pytest.mark.parametrize("tag_rows", [[], [{"name": "synthetic:1", "digest": "f" * 64}],
    [{"name": "synthetic:1", "digest": "a" * 64}] * 2])
def test_exact_model_digest_checked_without_pull_or_generation(monkeypatch, tag_rows):
    client = runner.OllamaClient(config())
    calls = []

    def request(method, path, body=None):
        calls.append((method, path))
        return {"version": "0.6.7"} if path == "/api/version" else {"models": tag_rows}

    monkeypatch.setattr(client, "request", request)
    with pytest.raises(runner.AuditError, match="model_pin_mismatch"):
        client.verify(version=True)
    assert calls == [("GET", "/api/version"), ("GET", "/api/tags")]


def test_recipes_share_definitions_but_have_distinct_order_and_request_hash():
    first, second = runner.SYSTEM_PROMPTS.values()
    assert first != second
    for definition in (runner._ASSISTANCE, runner._UTILITY):
        assert definition in first and definition in second
    p = packet()
    case, panel = p["cases"][0], p["cases"][0]["panels"][0]
    a, estimate = runner.make_request(case, panel, config())
    b, other_estimate = runner.make_request(case, panel, config() | {"recipe": "utility_first"})
    assert runner.sha(runner.canonical(a)) != runner.sha(runner.canonical(b))
    assert estimate == other_estimate and estimate + 192 <= 8192


def test_unfrozen_actual_scope_cannot_use_synthetic_packet(inputs):
    freeze, save, _ = inputs
    freeze["scope"] = "verified_a156"
    freeze["qualification_sha256"] = "d" * 64
    with pytest.raises(runner.AuditError, match="actual_authority"):
        runner.run_audit(**save(), client=FakeClient())
