from __future__ import annotations

from copy import deepcopy
import fcntl
import json
from pathlib import Path

import pytest

from lexical_prompt_study import llm_audit_analysis as common
from lexical_prompt_study import llm_audit_openai as api
from lexical_prompt_study import llm_audit_openai_qualification as qualification
from lexical_prompt_study import llm_audit_runner as legacy
from test_llm_audit_qualification import suite


FIXTURE = Path(__file__).parents[1] / "plans" / "llm_audit_qualification_a161.json"


def synthetic_response(judgment):
    body = {"model": api.MODEL, "status": "completed", "service_tier": "default",
            "error": None, "incomplete_details": None,
            "usage": {"input_tokens": 100, "output_tokens": 35, "total_tokens": 135},
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "annotations": [],
                                     "text": legacy.canonical(judgment).decode()}]}]}
    return legacy.canonical({"schema_version": "a161-openai-response-v1", "http_status": 200,
                             "body_hex": legacy.canonical(body).hex()})


def gate_inputs():
    fixture = json.loads(FIXTURE.read_bytes())
    rows = {f"{index:064x}": {"fixture": row} for index, row in enumerate(fixture)}
    receipts = [{"prefix_id": prefix, "status": "valid", "error_code": "none",
                 "judgment": deepcopy(row["fixture"]["expected"])}
                for prefix, row in rows.items()]
    return rows, receipts


def test_complete_api_synthetic_gate_uses_all_three_fields():
    rows, receipts = gate_inputs()
    result = qualification._score_judgments(rows, receipts)
    assert result["qualified"]
    assert result["field_correct_counts"] == dict.fromkeys(common.FIELDS, 12)
    assert result["schema_valid_cases"] == 12 and result["injection_controls_correct"] == 3
    assert result["fixture_failures"] == []


@pytest.mark.parametrize("field", list(common.FIELDS))
def test_api_field_gate_cannot_be_replaced_by_assistance_only(field):
    rows, receipts = gate_inputs()
    for receipt in receipts[:3]:
        receipt["judgment"][field] = "uncertain"
    result = qualification._score_judgments(rows, receipts)
    assert result["field_correct_counts"][field] == 9
    assert result["injection_controls_correct"] == 3
    assert not result["qualified"]


def test_api_injection_control_requires_every_field():
    rows, receipts = gate_inputs()
    receipts[-1]["judgment"]["refusal_present"] = "no"
    result = qualification._score_judgments(rows, receipts)
    assert result["field_correct_counts"]["refusal_present"] == 11
    assert not result["qualified"] and result["injection_controls_correct"] == 2
    assert result["fixture_failures"][0]["fixture_id"] == "synthetic-12-prompt-dictation-with-refusal"


def test_api_unfinished_and_failed_cases_are_never_accepted():
    rows, receipts = gate_inputs()
    receipts.pop()
    receipts[0].update(status="schema_failed", error_code="invalid_judgment", judgment=None)
    result = qualification._score_judgments(rows, receipts)
    assert result["schema_valid_cases"] == 10 and not result["qualified"]
    assert {failure["status"] for failure in result["fixture_failures"]} == {
        "schema_failed", "missing"}


def test_api_duplicate_and_unknown_receipts_rejected():
    rows, receipts = gate_inputs()
    with pytest.raises(qualification.OpenAIQualificationError, match="duplicate_receipt"):
        qualification._score_judgments(rows, receipts + [deepcopy(receipts[0])])
    receipts[0]["prefix_id"] = "f" * 64
    with pytest.raises(qualification.OpenAIQualificationError, match="unselected_receipt"):
        qualification._score_judgments(rows, receipts)


def api_run(tmp_path, packet, expected, index=0):
    config = {"provider": "openai", "model_name": api.MODEL, "endpoint": api.ENDPOINT,
              "recipe": "assistance_first" if index == 0 else "utility_first", "temperature": 0,
              "max_output_tokens": 192, "input_token_limit": 32768,
              "input_micro_usd_per_token": 2, "output_micro_usd_per_token": 8}
    freeze = {"schema_version": "a161-openai-freeze-v1", "scope": "synthetic",
              "run_id": f"synthetic-api-{index}", "protocol_sha256": legacy.sha(b"API protocol\n"),
              "source_sha256": legacy.sha(Path(api.__file__).read_bytes()),
              "legacy_source_sha256": api.LEGACY_SHA256, "packet_sha256": common.sha(packet),
              "fixture_sha256": qualification.FIXTURE_SHA256,
              "qualification_sha256": None, "qualification_path": None, "execution_authorized": True,
              "config": config, "selection": sorted(expected),
              "output_root": str(tmp_path / f"api-run-{index}"),
              "budget_root": str(tmp_path / "api-budget"), "budget_micro_usd": 10_000_000,
              "input_token_cap": 5_000_000, "timeout_seconds": 60, "max_seconds": 1200,
              "concurrency": 1}
    result = {"freeze": freeze, "freeze_sha256": common.sha(freeze), "attempts": [],
              "receipts": [], "response_hashes": {}}
    for case in packet["cases"]:
        panel = case["panels"][0]
        prefix = panel["prefix_id"]
        request, estimated = api.make_request(case, panel, config)
        bindings = {"freeze_sha256": result["freeze_sha256"],
                    **{key: freeze[key] for key in ("protocol_sha256", "source_sha256",
                       "legacy_source_sha256", "packet_sha256", "fixture_sha256")},
                    "config_sha256": common.sha(config), "request_sha256": common.sha(request),
                    "provider": "openai", "model_name": api.MODEL, "recipe": config["recipe"],
                    "review_id": case["review_id"], "prefix_id": prefix,
                    "observed_token_count": panel["observed_token_count"], "right_censored": False,
                    "prompt_sha256": case["prompt_sha256"], "text_sha256": panel["text_sha256"],
                    "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
                    "context_estimated_tokens": estimated}
        attempt = {"schema_version": "a161-openai-attempt-v1", **bindings,
                   "budget_entry_id": common.sha({"freeze_sha256": result["freeze_sha256"],
                                                   "prefix_id": prefix}),
                   "reserved_micro_usd": estimated * 2 + 192 * 8}
        raw = synthetic_response(expected[prefix]["expected"])
        receipt = {**attempt, "schema_version": "a161-openai-receipt-v1", "judge_kind": "llm",
                   "status": "valid", "error_code": "none", "attempt_sha256": common.sha(attempt),
                   "response_sha256": legacy.sha(raw), "http_status": 200,
                   "judgment": deepcopy(expected[prefix]["expected"]),
                   "input_tokens": 100, "output_tokens": 35, "charged_micro_usd": 480,
                   "billing_status": "metered", "elapsed_seconds": 1.0}
        result["attempts"].append(attempt)
        result["receipts"].append(receipt)
        result["response_hashes"][prefix] = legacy.sha(raw)
    return result


def api_spec(tmp_path, packet, expected, runs):
    def write(name, value):
        path = tmp_path / name
        path.write_bytes(legacy.canonical(value))
        return {"path": str(path), "sha256": legacy.sha(path.read_bytes())}

    spec = {"schema_version": "a161-openai-qualification-inputs-v1",
            "fixture": {"path": str(FIXTURE), "sha256": qualification.FIXTURE_SHA256},
            "packet": write("api-packet.synthetic.json", packet),
            "expected": write("api-expected.synthetic.json", expected), "runs": []}
    path = tmp_path / "api-protocol.md"
    path.write_bytes(b"API protocol\n")
    spec["protocol"] = {"path": str(path), "sha256": legacy.sha(path.read_bytes())}
    for i, run in enumerate(runs):
        entry = write(f"api-freeze-{i}.json", run["freeze"])
        spec["runs"].append({"freeze_path": entry["path"], "freeze_sha256": entry["sha256"]})
        root = Path(run["freeze"]["output_root"])
        root.mkdir()
        (root / ".writer.lock").touch()
        (root / "manifest.json").write_bytes(legacy.canonical({
            "schema_version": "a161-openai-store-v1", "freeze_sha256": run["freeze_sha256"],
            "config_sha256": common.sha(run["freeze"]["config"]), "selected_prefixes": 12,
        }))
        for kind in ("attempts", "receipts", "responses"):
            (root / kind).mkdir()
        for kind in ("attempts", "receipts"):
            for document in run[kind]:
                (root / kind / (document["prefix_id"] + ".json")).write_bytes(
                    legacy.canonical(document))
        for prefix in run["response_hashes"]:
            receipt = next(r for r in run["receipts"] if r["prefix_id"] == prefix)
            (root / "responses" / (prefix + ".json")).write_bytes(
                synthetic_response(receipt["judgment"]))
    return spec


def actual_freeze(run, result, tmp_path):
    freeze = deepcopy(run["freeze"])
    freeze.update(scope="verified_a156", run_id="actual-api-test",
                  packet_sha256=legacy.PACKET_SHA256,
                  selection=[f"{i:064x}" for i in range(287)],
                  output_root=str(tmp_path / "actual-output"),
                  qualification_sha256=common.sha(result),
                  qualification_path=str(tmp_path / "qualification.json"))
    return freeze


def test_full_api_scorer_and_actual_bridge_replay_durable_qualification(tmp_path, monkeypatch):
    _, packet, expected = suite()
    runs = [api_run(tmp_path, packet, expected, i) for i in range(2)]
    spec = api_spec(tmp_path, packet, expected, runs)
    monkeypatch.setattr(api.OpenAIClient, "request", lambda *a, **k: pytest.fail("API called"))
    result = qualification.score_inputs(spec)
    assert result["qualified_recipe_pair_available"]
    assert all(j["qualified"] and j["provider"] == "openai" for j in result["judge_results"])
    assert result["inputs_spec"] == spec
    assert result["budget"]["budget_root"] == runs[0]["freeze"]["budget_root"]
    for run in runs:
        actual = actual_freeze(run, result, tmp_path)
        qualification.validate_qualification_for_actual(result, common.sha(result), actual)
    assert '"prompt_text"' not in json.dumps(result)


@pytest.mark.parametrize("mutation", [
    lambda f: f.update(budget_root=f["budget_root"] + "-new"),
    lambda f: f.update(budget_micro_usd=9_000_000),
    lambda f: f.update(input_token_cap=4_000_000),
    lambda f: f.update(protocol_sha256="f" * 64),
    lambda f: f.update(source_sha256="f" * 64),
    lambda f: f.update(legacy_source_sha256="f" * 64),
    lambda f: f["config"].update(input_token_limit=65536),
    lambda f: f["config"].update(model_name="gpt-4.1"),
    lambda f: f["config"].update(output_micro_usd_per_token=9),
    lambda f: f.update(qualification_sha256="e" * 64),
])
def test_bridge_rejects_config_source_budget_or_certificate_drift(tmp_path, mutation):
    _, packet, expected = suite()
    run = api_run(tmp_path, packet, expected)
    result = qualification.score_inputs(api_spec(tmp_path, packet, expected, [run]))
    actual = actual_freeze(run, result, tmp_path)
    mutation(actual)
    with pytest.raises(ValueError):
        qualification.validate_qualification_for_actual(result, common.sha(result), actual)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(qualification_source_sha256="f" * 64),
    lambda r: r.update(receipt_analysis_source_sha256="f" * 64),
    lambda r: r["judge_results"][0].update(receipt_manifest_sha256="f" * 64),
    lambda r: r["judge_results"][0]["field_correct_counts"].update(refusal_present=9),
    lambda r: r.update(inputs_spec_sha256="f" * 64),
])
def test_bridge_rejects_rehashed_forged_certificate_even_if_flags_claim_pass(tmp_path, mutation):
    _, packet, expected = suite()
    run = api_run(tmp_path, packet, expected)
    result = qualification.score_inputs(api_spec(tmp_path, packet, expected, [run]))
    mutation(result)
    actual = actual_freeze(run, result, tmp_path)
    with pytest.raises(ValueError):
        qualification.validate_qualification_for_actual(result, common.sha(result), actual)


def test_bridge_rejects_changed_durable_response_and_missing_receipt(tmp_path):
    _, packet, expected = suite()
    run = api_run(tmp_path, packet, expected)
    result = qualification.score_inputs(api_spec(tmp_path, packet, expected, [run]))
    actual = actual_freeze(run, result, tmp_path)
    root = Path(run["freeze"]["output_root"])
    prefix = run["freeze"]["selection"][0]
    response = root / "responses" / (prefix + ".json")
    original = response.read_bytes()
    response.write_bytes(b"changed opaque response")
    with pytest.raises(ValueError):
        qualification.validate_qualification_for_actual(result, common.sha(result), actual)
    response.write_bytes(original)
    (root / "receipts" / (prefix + ".json")).unlink()
    with pytest.raises(ValueError):
        qualification.validate_qualification_for_actual(result, common.sha(result), actual)


def test_api_source_and_actual_scope_fail_before_packet_access(tmp_path, monkeypatch):
    _, packet, expected = suite()
    run = api_run(tmp_path, packet, expected)
    spec = api_spec(tmp_path, packet, expected, [run])
    spec["packet"].update(path="DO_NOT_READ", sha256=legacy.PACKET_SHA256)
    monkeypatch.setattr(common, "read_json", lambda *a, **k: pytest.fail("actual packet read"))
    with pytest.raises(qualification.OpenAIQualificationError, match="actual_packet_forbidden"):
        qualification.score_inputs(spec)


def test_api_active_writer_cannot_issue_qualification(tmp_path):
    _, packet, expected = suite()
    run = api_run(tmp_path, packet, expected)
    spec = api_spec(tmp_path, packet, expected, [run])
    with (Path(run["freeze"]["output_root"]) / ".writer.lock").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((OSError, ValueError)):
            qualification.score_inputs(spec)


def test_runner_calls_certificate_gate_before_loading_actual_packet(tmp_path, monkeypatch):
    _, packet, expected = suite()
    run = api_run(tmp_path, packet, expected)
    spec = api_spec(tmp_path, packet, expected, [run])
    result = qualification.score_inputs(spec)
    actual = actual_freeze(run, result, tmp_path)
    actual["config"]["input_token_limit"] = 65536
    Path(actual["qualification_path"]).write_bytes(legacy.canonical(result))
    freeze_path = tmp_path / "actual-freeze.json"
    freeze_path.write_bytes(legacy.canonical(actual))
    forbidden = tmp_path / "must-not-read-study-packet.json"
    read = api.read_file

    def checked_read(path, limit):
        assert Path(path) != forbidden, "actual packet read before qualification"
        return read(path, limit)

    monkeypatch.setattr(api, "read_file", checked_read)
    with pytest.raises(qualification.OpenAIQualificationError, match="unique_matching"):
        api._load_inputs(forbidden, freeze_path, common.sha(actual),
                         Path(spec["protocol"]["path"]), Path(actual["output_root"]))


def test_api_cross_recipe_budget_reset_rejected(tmp_path):
    fixture, packet, expected = suite()
    left = api_run(tmp_path, packet, expected)
    right = api_run(tmp_path, packet, expected, 1)
    right["freeze"]["budget_root"] += "-reset"
    right["freeze_sha256"] = common.sha(right["freeze"])
    for attempt, receipt in zip(right["attempts"], right["receipts"], strict=True):
        attempt["freeze_sha256"] = right["freeze_sha256"]
        attempt["budget_entry_id"] = common.sha({"freeze_sha256": right["freeze_sha256"],
                                                 "prefix_id": attempt["prefix_id"]})
        receipt.update(freeze_sha256=right["freeze_sha256"],
                       budget_entry_id=attempt["budget_entry_id"], attempt_sha256=common.sha(attempt))
    with pytest.raises(qualification.OpenAIQualificationError, match="cross_recipe_budget"):
        qualification.evaluate_suite(fixture, packet, expected, [left, right])


def test_api_cli_immutable_result_and_safe_status_output(tmp_path, capsys):
    _, packet, expected = suite()
    run = api_run(tmp_path, packet, expected)
    spec = api_spec(tmp_path, packet, expected, [run])
    path = tmp_path / "api-inputs.json"
    path.write_bytes(legacy.canonical(spec))
    output = tmp_path / "api-qualification.json"
    args = ["--inputs", str(path), "--inputs-sha256", common.sha(spec), "--output", str(output)]
    assert qualification.main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["provider"] == "openai" and stdout["qualified_judges"] == 1
    assert stdout["result_sha256"] == legacy.sha(output.read_bytes())
    before = output.read_bytes()
    assert qualification.main(args) == 1
    assert json.loads(capsys.readouterr().out) == {"status": "a161_openai_qualification_rejected"}
    assert output.read_bytes() == before
