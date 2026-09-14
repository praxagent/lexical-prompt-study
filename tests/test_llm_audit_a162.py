"""A162 instrument/source/accounting tests: invented evidence and fake API only."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lexical_prompt_study import llm_audit_a162 as runner
from lexical_prompt_study import llm_audit_openai as provider


JUDGMENT = {"disallowed_assistance": "no", "safe_task_completion": "complete", "refusal_present": "no"}


def packet(count=12):
    cases = []
    for index in range(count):
        review_id = f"{index + 1:024x}"
        text = f"Invented answer {index}."
        prompt = f"Invented prompt {index}; evaluate only this case."
        panel = {"text": text, "text_sha256": runner.sha(text.encode()),
                 "generated_token_ids_sha256": runner.sha(str(index).encode()),
                 "observed_token_count": 8, "right_censored": False}
        panel["prefix_id"] = runner.sha(runner.canonical({"review_id": review_id,
            "observed_token_count": 8, "right_censored": False,
            "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
            "generated_text_sha256": panel["text_sha256"]}))
        cases.append({"review_id": review_id, "prompt_text": prompt,
            "prompt_sha256": runner.sha(prompt.encode()),
            "prompt_rendering": "exact_chat_template_serialization", "panels": [panel]})
    return {"schema_version": "a156-display-v1", "scope": "synthetic", "cases": cases,
        "provenance": {name: "b" * 64 for name in (
            "request_sha256", "bundle_sha256", "plan_sha256", "builder_source_sha256")}}


class FakeClient:
    def __init__(self, error=None, content=None):
        self.calls = []
        self.error = error
        self.content = runner.canonical(JUDGMENT).decode() if content is None else content

    def request(self, body):
        self.calls.append(deepcopy(body))
        if self.error is not None:
            raise self.error
        return 200, runner.canonical({"model": provider.MODEL, "service_tier": "default",
            "status": "completed", "error": None, "incomplete_details": None,
            "usage": {"input_tokens": 400, "output_tokens": 40, "total_tokens": 440},
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": self.content, "annotations": []}]}]})


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "BUDGET_ROOT", str(tmp_path / "shared-a161-budget"))
    rubric = {"assistance_first": "A162 synthetic safety-first rubric. Judge visible evidence only.",
              "utility_first": "A162 synthetic utility-first rubric. Judge visible evidence only."}
    rubric_path = tmp_path / "rubric.json"
    rubric_path.write_bytes(runner.canonical(rubric))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Prospective synthetic A162 protocol.")
    holdout = tmp_path / "holdout.json"
    holdout.write_bytes(runner.canonical({"synthetic_holdout_fixture": 24}))
    development = Path(__file__).parents[1] / "plans" / "llm_audit_qualification_a161.json"
    cfg = {"instrument_id": "a162", "rubric_sha256": runner.sha(rubric_path.read_bytes()),
        "provider": "openai", "model_name": provider.MODEL, "endpoint": provider.ENDPOINT,
        "recipe": "assistance_first", "temperature": 0, "max_output_tokens": 192,
        "input_token_limit": 32768, "input_micro_usd_per_token": 2, "output_micro_usd_per_token": 8}
    p = packet()
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(runner.canonical(p))
    freeze = {"schema_version": "a162-freeze-v1", "instrument_id": "a162", "stage": "development",
        "scope": "synthetic", "run_id": "a162-test-development",
        "protocol_sha256": runner.sha(protocol.read_bytes()),
        "source_sha256": runner.sha(Path(runner.__file__).read_bytes()),
        "engine_source_sha256": runner.ENGINE_SHA256, "legacy_source_sha256": runner.LEGACY_SHA256,
        "packet_sha256": runner.sha(packet_path.read_bytes()), "fixture_sha256": runner.DEVELOPMENT_FIXTURE_SHA256,
        "rubric_path": str(rubric_path), "rubric_sha256": cfg["rubric_sha256"],
        "holdout_fixture_path": str(holdout), "holdout_fixture_sha256": runner.sha(holdout.read_bytes()),
        "development_fixture_path": str(development), "development_fixture_sha256": runner.DEVELOPMENT_FIXTURE_SHA256,
        "qualification_sha256": None, "qualification_path": None, "execution_authorized": True,
        "config": cfg, "selection": sorted(c["panels"][0]["prefix_id"] for c in p["cases"]),
        "output_root": str(tmp_path / "output"), "budget_root": runner.BUDGET_ROOT,
        "budget_protocol_sha256": runner.BUDGET_PROTOCOL_SHA256, "budget_micro_usd": 10_000_000,
        "input_token_cap": 5_000_000, "timeout_seconds": 5, "max_seconds": 60, "concurrency": 1}
    budget_root = provider._private_directory(Path(runner.BUDGET_ROOT))
    fd = provider._lock(budget_root)
    try:
        provider.BudgetLedger(budget_root, {**freeze, "protocol_sha256": runner.BUDGET_PROTOCOL_SHA256})
    finally:
        provider.os.close(fd)
    freeze_path = tmp_path / "freeze.json"

    def save():
        freeze_path.write_bytes(runner.canonical(freeze))
        return dict(packet_path=packet_path, freeze_path=freeze_path,
            expected_freeze_sha256=runner.sha(freeze_path.read_bytes()), protocol_path=protocol,
            output_root=Path(freeze["output_root"]))

    return freeze, save, p, rubric


def test_exact_new_rubric_request_hash_and_source_identity(inputs, capsys):
    freeze, save, p, rubric = inputs
    args = save()
    old_sources = {Path(module.__file__): Path(module.__file__).read_bytes() for module in (provider, runner.legacy)}
    client = FakeClient()
    result = runner.run_audit(**args, client=client)
    assert result["counts"]["valid"] == 12 and result["instrument_id"] == "a162"
    assert result["ledger_charged_micro_usd"] == 12 * 1120
    assert all(body["input"][0]["content"] == rubric["assistance_first"] for body in client.calls)
    requests = {runner.sha(runner.canonical(body)): body for body in client.calls}
    for path in (args["output_root"] / "receipts").glob("*.json"):
        receipt = runner.decode(path.read_bytes())
        assert receipt["freeze_sha256"] == args["expected_freeze_sha256"]
        assert receipt["source_sha256"] == freeze["source_sha256"]
        assert receipt["protocol_sha256"] == freeze["protocol_sha256"] != runner.BUDGET_PROTOCOL_SHA256
        assert receipt["config_sha256"] == runner.sha(runner.canonical(freeze["config"]))
        assert receipt["request_sha256"] in requests
        assert requests[receipt["request_sha256"]]["input"][0]["content"] == rubric["assistance_first"]
        assert "Invented prompt" not in path.read_text()
    old_cfg = {key: freeze["config"][key] for key in provider.CONFIG_FIELDS}
    old_body, _ = provider.make_request(p["cases"][0], p["cases"][0]["panels"][0], old_cfg)
    new_body, _ = runner.make_request(p["cases"][0], p["cases"][0]["panels"][0], freeze["config"], rubric)
    assert runner.sha(runner.canonical(old_body)) != runner.sha(runner.canonical(new_body))
    assert all(path.read_bytes() == raw for path, raw in old_sources.items())
    assert capsys.readouterr() == ("", "")


def test_fresh_evidence_envelope_has_no_other_cases_or_judge_metadata(inputs):
    freeze, _, p, rubric = inputs
    case = p["cases"][0]
    body, estimated = runner.make_request(case, case["panels"][0], freeze["config"], rubric)
    evidence = runner.decode(body["input"][1]["content"])
    assert evidence == {"model_visible_prompt": case["prompt_text"],
        "observed_response_prefix": case["panels"][0]["text"], "observed_token_count": 8, "right_censored": False}
    assert "Invented prompt 1;" not in runner.canonical(body).decode()
    assert body["service_tier"] == "default" and body["store"] is False and body["max_output_tokens"] == 192
    assert estimated == len(runner.canonical(body)) + 1024


def test_completed_resume_is_immutable_and_never_requeries(inputs):
    _, save, _, _ = inputs
    args = save()
    result = runner.run_audit(**args, client=FakeClient())
    files = {path: path.read_bytes() for path in args["output_root"].rglob("*.json")}
    client = FakeClient(error=AssertionError("must not retry"))
    assert runner.run_audit(**args, client=client) == result and not client.calls
    assert {path: path.read_bytes() for path in files} == files


def test_interrupted_attempt_preserves_cumulative_reservation_without_retry(inputs):
    _, save, _, _ = inputs
    args = save()
    with pytest.raises(KeyboardInterrupt):
        runner.run_audit(**args, client=FakeClient(error=KeyboardInterrupt()))
    client = FakeClient(error=AssertionError("must not retry"))
    result = runner.run_audit(**args, client=client)
    assert not client.calls and result["interrupted_attempts"] == 1 and result["ledger_charged_micro_usd"] > 0


def test_old_unresolved_spend_blocks_new_instrument_before_call(inputs):
    freeze, save, p, _ = inputs
    args = save()
    root = Path(freeze["budget_root"])
    ledger = provider.BudgetLedger(root, {**freeze, "protocol_sha256": runner.BUDGET_PROTOCOL_SHA256})
    bindings = provider._bindings(freeze, "a" * 64, "b" * 64,
                                  p["cases"][0], p["cases"][0]["panels"][0], {"prior": True}, 4_999_000)
    entry_id = runner.sha(runner.canonical({"freeze_sha256": bindings["freeze_sha256"], "prefix_id": bindings["prefix_id"]}))
    attempt = {"schema_version": "a161-openai-attempt-v1", **bindings,
               "budget_entry_id": entry_id, "reserved_micro_usd": 4_999_000 * 2 + 192 * 8}
    assert ledger.reserve(attempt, args["output_root"].parent / "prior-a161")
    client = FakeClient(error=AssertionError("budget must block"))
    result = runner.run_audit(**args, client=client)
    assert not client.calls and result["stop_reason"] == "cost_or_input_budget"
    assert result["ledger_charged_micro_usd"] == attempt["reserved_micro_usd"]


@pytest.mark.parametrize("field,value", [
    ("run_id", "a161-reused"), ("engine_source_sha256", "f" * 64), ("legacy_source_sha256", "f" * 64),
    ("budget_root", "/tmp/new-budget"), ("budget_micro_usd", 9_000_000),
    ("budget_protocol_sha256", "f" * 64), ("input_token_cap", 4_000_000),
    ("development_fixture_sha256", "f" * 64), ("fixture_sha256", "f" * 64),
    ("stage", "unknown"), ("scope", "verified_a156"), ("selection", ["a" * 64]),
])
def test_instrument_freeze_rejects_wrong_lineage_stage_budget_and_selection(inputs, field, value):
    freeze, _, _, _ = inputs
    freeze[field] = value
    with pytest.raises(runner.AuditError):
        runner.validate_freeze(freeze)


@pytest.mark.parametrize("kind", ["rubric", "holdout_fixture", "source", "protocol"])
def test_modified_frozen_inputs_rejected_before_packet_read(inputs, monkeypatch, kind):
    freeze, save, _, _ = inputs
    if kind in {"rubric", "holdout_fixture"}:
        Path(freeze[kind + "_path"]).write_text("changed")
    elif kind == "source":
        freeze["source_sha256"] = "e" * 64
    else:
        freeze["protocol_sha256"] = "e" * 64
    args = save()
    original = runner.read_file
    reads = []

    def recording_read(path, limit):
        reads.append(path)
        return original(path, limit)

    monkeypatch.setattr(runner, "read_file", recording_read)
    with pytest.raises(runner.AuditError):
        runner.run_audit(**args, client=FakeClient(error=AssertionError("must not infer")))
    assert args["packet_path"] not in reads


@pytest.mark.parametrize("stage", ["holdout", "actual"])
def test_concrete_stage_gate_precedes_selected_packet_access(inputs, tmp_path, monkeypatch, stage):
    freeze, save, _, _ = inputs
    certificate = tmp_path / "certificate.json"
    certificate.write_bytes(runner.canonical({"synthetic": True}))
    freeze.update(stage=stage, scope="verified_a156" if stage == "actual" else "synthetic",
        fixture_sha256=freeze["holdout_fixture_sha256"], qualification_path=str(certificate),
        qualification_sha256=runner.sha(certificate.read_bytes()),
        selection=sorted(runner.sha(str(i).encode()) for i in range(287 if stage == "actual" else 24)))
    if stage == "actual":
        freeze["packet_sha256"] = runner.PACKET_SHA256
    args = save()
    reads = []
    original = runner.read_file
    gates = []

    def recording_read(path, limit):
        reads.append(path)
        return original(path, limit)

    def gate(*arguments):
        gates.append(stage)
        raise runner.AuditError("synthetic_gate_rejected")

    gate_module = SimpleNamespace(validate_development_for_holdout=gate, validate_qualification_for_actual=gate)
    monkeypatch.setitem(sys.modules, "lexical_prompt_study.llm_audit_a162_qualification", gate_module)
    monkeypatch.setattr(sys.modules["lexical_prompt_study"], "llm_audit_a162_qualification", gate_module, raising=False)
    monkeypatch.setattr(runner, "read_file", recording_read)
    with pytest.raises(runner.AuditError, match="synthetic_gate_rejected"):
        runner.run_audit(**args, client=FakeClient(error=AssertionError("must not infer")))
    assert gates == [stage] and args["packet_path"] not in reads


def test_holdout_uses_all_24_fresh_cases_after_development_gate(inputs, tmp_path, monkeypatch):
    freeze, save, _, _ = inputs
    p = packet(24)
    certificate = tmp_path / "development-result.json"
    certificate.write_bytes(runner.canonical({"synthetic_development_pass": True}))
    args = save()
    args["packet_path"].write_bytes(runner.canonical(p))
    freeze.update(stage="holdout", fixture_sha256=freeze["holdout_fixture_sha256"],
        packet_sha256=runner.sha(args["packet_path"].read_bytes()),
        qualification_path=str(certificate), qualification_sha256=runner.sha(certificate.read_bytes()),
        selection=sorted(c["panels"][0]["prefix_id"] for c in p["cases"]))
    gate_calls = []
    gate_module = SimpleNamespace(validate_development_for_holdout=lambda *a: gate_calls.append(a))
    monkeypatch.setitem(sys.modules, "lexical_prompt_study.llm_audit_a162_qualification", gate_module)
    monkeypatch.setattr(sys.modules["lexical_prompt_study"], "llm_audit_a162_qualification", gate_module, raising=False)
    client = FakeClient()
    result = runner.run_audit(**save(), client=client)
    assert len(gate_calls) == 1 and len(client.calls) == 24 and result["counts"]["valid"] == 24


def test_context_overflow_refuses_without_truncation_or_attempt(inputs):
    freeze, save, _, _ = inputs
    freeze["config"]["input_token_limit"] = 1024
    args = save()
    with pytest.raises(runner.AuditError, match="context_budget_exceeded"):
        runner.run_audit(**args, client=FakeClient(error=AssertionError("overflow")))
    assert not args["output_root"].exists()


def test_malformed_answer_retains_frozen_native_failure_semantics(inputs):
    _, save, _, _ = inputs
    args = save()
    result = runner.run_audit(**args, client=FakeClient(content='{"rationale":"PRIVATE-PROSE"}'))
    assert result["counts"]["schema_failed"] == 12
    for path in (args["output_root"] / "receipts").glob("*.json"):
        row = runner.decode(path.read_bytes())
        assert row["error_code"] == "invalid_judgment" and row["judgment"] is None
        assert "PRIVATE-PROSE" not in path.read_text()
