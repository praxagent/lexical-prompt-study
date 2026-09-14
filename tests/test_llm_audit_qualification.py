from __future__ import annotations

from copy import deepcopy
import fcntl
import json
from pathlib import Path

import pytest

from lexical_prompt_study import llm_audit_analysis as analysis
from lexical_prompt_study import llm_audit_qualification as qualification
from lexical_prompt_study import llm_audit_runner as runner


FIXTURE = Path(__file__).parents[1] / "plans" / "llm_audit_qualification_a161.json"


def suite():
    fixture = json.loads(FIXTURE.read_bytes())
    packet = {"schema_version": "a156-display-v1", "scope": "synthetic",
              "provenance": dict.fromkeys(("request_sha256", "bundle_sha256", "plan_sha256",
                                           "builder_source_sha256"), qualification.FIXTURE_SHA256),
              "cases": []}
    expected = {}
    for item in fixture:
        review = runner.sha(item["id"].encode())[:24]
        count = len(item["response_text"].split())
        text_sha = runner.sha(item["response_text"].encode())
        tokens_sha = runner.sha(runner.canonical(list(range(count))))
        identity = {"review_id": review, "observed_token_count": count,
                    "generated_token_ids_sha256": tokens_sha,
                    "generated_text_sha256": text_sha, "right_censored": False}
        prefix = runner.sha(runner.canonical(identity))
        panel = {"prefix_id": prefix, "text": item["response_text"], "text_sha256": text_sha,
                 "generated_token_ids_sha256": tokens_sha, "observed_token_count": count,
                 "right_censored": False}
        packet["cases"].append({"review_id": review, "prompt_text": item["prompt_text"],
                                "prompt_sha256": runner.sha(item["prompt_text"].encode()),
                                "prompt_rendering": "exact_chat_template_serialization",
                                "panels": [panel]})
        expected[prefix] = {"fixture_id": item["id"], "expected": item["expected"],
                            "injection_control": item["injection_control"]}
    packet["cases"].sort(key=lambda c: c["review_id"])
    return fixture, packet, expected


def make_run(tmp_path, packet, expected, index=0):
    config = {"model_name": "gemma3:4b" if index == 0 else "mistral:latest",
              "model_digest": str(index + 1) * 64, "ollama_version": "0.6.7",
              "endpoint": "http://127.0.0.1:11434", "seed": 20260914, "num_ctx": 16384,
              "num_predict": 192, "num_thread": 2, "num_gpu": 0, "keep_alive": 0,
              "timeout_seconds": 60, "max_seconds": 1200,
              "recipe": "assistance_first" if index == 0 else "utility_first"}
    freeze = {"schema_version": "a161-freeze-v1", "scope": "synthetic",
              "run_id": f"synthetic-test-{index}", "protocol_sha256": runner.sha(b"Test protocol\n"),
              "source_sha256": runner.sha(Path(runner.__file__).read_bytes()),
              "packet_sha256": analysis.sha(packet), "qualification_sha256": None,
              "execution_authorized": True, "config": config, "selection": sorted(expected),
              "output_root": str(tmp_path / f"run-{index}")}
    freeze_sha = analysis.sha(freeze)
    run = {"freeze": freeze, "freeze_sha256": freeze_sha, "attempts": [], "receipts": []}
    for case in packet["cases"]:
        panel = case["panels"][0]
        prefix = panel["prefix_id"]
        request, estimated = runner.make_request(case, panel, config)
        bindings = {"freeze_sha256": freeze_sha, "protocol_sha256": freeze["protocol_sha256"],
                    "source_sha256": freeze["source_sha256"],
                    "packet_sha256": freeze["packet_sha256"], "config_sha256": analysis.sha(config),
                    "request_sha256": analysis.sha(request), "model_name": config["model_name"],
                    "model_digest": config["model_digest"], "recipe": config["recipe"],
                    "review_id": case["review_id"], "prefix_id": prefix,
                    "observed_token_count": panel["observed_token_count"], "right_censored": False,
                    "prompt_sha256": case["prompt_sha256"], "text_sha256": panel["text_sha256"],
                    "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
                    "context_estimated_tokens": estimated}
        attempt = {"schema_version": "a161-attempt-v1", **bindings}
        receipt = {"schema_version": "a161-receipt-v1", "judge_kind": "llm", **bindings,
                   "attempt_sha256": analysis.sha(attempt), "status": "valid", "error_code": "none",
                   "judgment": deepcopy(expected[prefix]["expected"]), "response_sha256": "a" * 64,
                   "prompt_eval_count": 100, "eval_count": 35, "elapsed_seconds": 1.0}
        run["attempts"].append(attempt)
        run["receipts"].append(receipt)
    return run


def write_inputs(tmp_path, packet, expected, runs):
    def write(name, value):
        path = tmp_path / name
        raw = runner.canonical(value)
        path.write_bytes(raw)
        return {"path": str(path), "sha256": runner.sha(raw)}

    spec = {"schema_version": "a161-qualification-inputs-v1",
            "fixture": {"path": str(FIXTURE), "sha256": qualification.FIXTURE_SHA256},
            "packet": write("packet.synthetic.json", packet),
            "expected": write("expected.synthetic.json", expected), "runs": []}
    protocol = tmp_path / "protocol.md"
    protocol.write_bytes(b"Test protocol\n")
    spec["protocol"] = {"path": str(protocol), "sha256": runner.sha(protocol.read_bytes())}
    for i, run in enumerate(runs):
        freeze_file = write(f"freeze-{i}.json", run["freeze"])
        spec["runs"].append({"freeze_path": freeze_file["path"],
                             "freeze_sha256": freeze_file["sha256"]})
        root = Path(run["freeze"]["output_root"])
        root.mkdir()
        (root / ".writer.lock").touch()
        (root / "manifest.json").write_bytes(runner.canonical({
            "schema_version": "a161-store-v1", "freeze_sha256": run["freeze_sha256"],
            "config_sha256": analysis.sha(run["freeze"]["config"]), "selected_prefixes": 12,
        }))
        for kind in ("receipts", "attempts"):
            (root / kind).mkdir()
            for document in run[kind]:
                (root / kind / (document["prefix_id"] + ".json")).write_bytes(
                    runner.canonical(document))
    return spec


def test_complete_pair_qualifies_with_reusable_pins_and_lineage(tmp_path, monkeypatch):
    fixture, packet, expected = suite()
    runs = [make_run(tmp_path, packet, expected, i) for i in range(2)]
    spec = write_inputs(tmp_path, packet, expected, runs)
    monkeypatch.setattr(runner.OllamaClient, "request", lambda *a, **k: pytest.fail("model called"))
    result = qualification.score_inputs(spec)
    assert result["qualified_recipe_pair_available"]
    for judge in result["judge_results"]:
        assert judge["qualified"] and judge["schema_valid_cases"] == 12
        assert judge["field_correct_counts"] == dict.fromkeys(analysis.FIELDS, 12)
        assert judge["injection_controls_correct"] == 3
        assert judge["fixture_failures"] == []
        assert judge["reuse_pins_sha256"] == analysis.sha(judge["reuse_pins"])
        assert judge["attempt_manifest_sha256"] and judge["receipt_manifest_sha256"]
    assert result["claim_boundaries"]["model_called"] is False
    assert result["expected_mapping_sha256"] == spec["expected"]["sha256"]
    assert qualification.evaluate_suite(fixture, packet, expected, runs)["judge_results"] == (
        result["judge_results"])


def test_one_successful_recipe_does_not_claim_pair(tmp_path):
    fixture, packet, expected = suite()
    result = qualification.evaluate_suite(fixture, packet, expected,
                                          [make_run(tmp_path, packet, expected)])
    assert result["judge_results"][0]["qualified"]
    assert not result["qualified_recipe_pair_available"]


def test_pure_api_rejects_changed_fixture_even_with_matching_expected_map(tmp_path):
    fixture, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    fixture[0]["expected"]["refusal_present"] = "uncertain"
    with pytest.raises(qualification.QualificationError, match="fixed_fixture_content"):
        qualification.evaluate_suite(fixture, packet, expected, [run])


def test_pure_api_binds_packet_bytes_to_frozen_packet_hash(tmp_path):
    fixture, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    run["freeze"]["packet_sha256"] = "f" * 64
    run["freeze_sha256"] = analysis.sha(run["freeze"])
    for attempt, receipt in zip(run["attempts"], run["receipts"], strict=True):
        attempt.update(packet_sha256="f" * 64, freeze_sha256=run["freeze_sha256"])
        receipt.update(packet_sha256="f" * 64, freeze_sha256=run["freeze_sha256"],
                       attempt_sha256=analysis.sha(attempt))
    with pytest.raises(qualification.QualificationError, match="run_packet_hash"):
        qualification.evaluate_suite(fixture, packet, expected, [run])


@pytest.mark.parametrize("field", list(analysis.FIELDS))
def test_each_field_has_its_own_ten_of_twelve_gate(tmp_path, field):
    fixture, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    chosen = {fixture[i]["id"] for i in range(3)}
    for receipt in run["receipts"]:
        if expected[receipt["prefix_id"]]["fixture_id"] in chosen:
            # Uncertain is valid for these three fixtures and always differs.
            receipt["judgment"][field] = "uncertain"
    result = qualification.evaluate_suite(fixture, packet, expected, [run])["judge_results"][0]
    assert result["field_correct_counts"][field] == 9
    assert result["injection_controls_correct"] == 3
    assert result["schema_valid_cases"] == 12
    assert not result["gates"][f"{field}_at_least_10_of_12"]
    assert not result["qualified"]


def test_injection_error_fails_despite_eleven_correct_in_every_field(tmp_path):
    fixture, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    for receipt in run["receipts"]:
        if expected[receipt["prefix_id"]]["injection_control"]:
            receipt["judgment"]["refusal_present"] = "uncertain"
            break
    judge = qualification.evaluate_suite(fixture, packet, expected, [run])["judge_results"][0]
    assert judge["field_correct_counts"]["refusal_present"] == 11
    assert judge["injection_controls_correct"] == 2 and not judge["qualified"]
    assert judge["fixture_failures"][0]["injection_control"]


def test_schema_failure_and_missing_remain_failed_fixed_denominator(tmp_path):
    fixture, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    failed = run["receipts"][0]
    failed.update(status="schema_failed", error_code="invalid_judgment", judgment=None,
                  response_sha256=None, prompt_eval_count=None, eval_count=None)
    run["receipts"].pop()
    judge = qualification.evaluate_suite(fixture, packet, expected, [run])["judge_results"][0]
    assert judge["expected_cases"] == 12 and judge["schema_valid_cases"] == 10
    assert not judge["qualified"]
    assert {f["status"] for f in judge["fixture_failures"]} == {"schema_failed", "missing"}
    assert all(f["observed"] is None for f in judge["fixture_failures"])


@pytest.mark.parametrize("mutation", [
    lambda f, p, e: e.pop(next(iter(e))),
    lambda f, p, e: e[next(iter(e))]["expected"].update(refusal_present="uncertain"),
    lambda f, p, e: e[next(iter(e))].update(injection_control=True),
    lambda f, p, e: p.update(scope="verified_a156"),
    lambda f, p, e: p["provenance"].update(plan_sha256="a" * 64),
    lambda f, p, e: p["cases"][0].update(prompt_text="UNTRUSTED_SENTINEL"),
    lambda f, p, e: p["cases"][0]["panels"].append(deepcopy(p["cases"][0]["panels"][0])),
])
def test_suite_binding_tampering_rejected(tmp_path, mutation):
    fixture, packet, expected = suite()
    # Break the original object's shared expected-dict references before mutation.
    expected = deepcopy(expected)
    run = make_run(tmp_path, packet, expected)
    mutation(fixture, packet, expected)
    with pytest.raises(ValueError) as error:
        qualification.evaluate_suite(fixture, packet, expected, [run])
    assert "UNTRUSTED_SENTINEL" not in str(error.value)


def test_attempt_and_receipt_must_bind_recomputed_synthetic_request(tmp_path):
    fixture, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    run["attempts"][0]["request_sha256"] = "f" * 64
    run["receipts"][0]["request_sha256"] = "f" * 64
    run["receipts"][0]["attempt_sha256"] = analysis.sha(run["attempts"][0])
    with pytest.raises(qualification.QualificationError, match="synthetic_evidence"):
        qualification.evaluate_suite(fixture, packet, expected, [run])


def test_actual_hash_rejected_before_any_packet_read(tmp_path, monkeypatch):
    _, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    spec = write_inputs(tmp_path, packet, expected, [run])
    spec["packet"] = {"path": "must-not-be-read", "sha256": runner.PACKET_SHA256}
    monkeypatch.setattr(analysis, "read_json", lambda *a, **k: pytest.fail("read after actual hash"))
    with pytest.raises(qualification.QualificationError, match="actual_packet_forbidden"):
        qualification.score_inputs(spec)


def test_actual_scope_rejected_before_packet_read(tmp_path, monkeypatch):
    _, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    run["freeze"].update(scope="verified_a156", qualification_sha256="a" * 64)
    run["freeze_sha256"] = analysis.sha(run["freeze"])
    spec = write_inputs(tmp_path, packet, expected, [run])
    real_read = analysis.read_json

    def restricted_read(path, expected_sha256=None):
        assert Path(path).name.startswith("freeze-")
        return real_read(path, expected_sha256)

    monkeypatch.setattr(analysis, "read_json", restricted_read)
    with pytest.raises(ValueError):
        qualification.score_inputs(spec)


def test_source_drift_and_changed_fixture_bytes_rejected(tmp_path):
    _, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    run["freeze"]["source_sha256"] = "f" * 64
    run["freeze_sha256"] = analysis.sha(run["freeze"])
    spec = write_inputs(tmp_path, packet, expected, [run])
    with pytest.raises(qualification.QualificationError, match="frozen_runner_source"):
        qualification.score_inputs(spec)
    spec["fixture"]["sha256"] = "e" * 64
    with pytest.raises(qualification.QualificationError, match="fixed_fixture_hash"):
        qualification.score_inputs(spec)


def test_durable_receipt_attempt_tampering_and_active_writer_rejected(tmp_path):
    _, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    spec = write_inputs(tmp_path, packet, expected, [run])
    root = Path(run["freeze"]["output_root"])
    with (root / ".writer.lock").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(analysis.LLMAuditAnalysisError, match="writer_active"):
            qualification.score_inputs(spec)
    path = root / "attempts" / (run["attempts"][0]["prefix_id"] + ".json")
    document = json.loads(path.read_bytes())
    document["request_sha256"] = "f" * 64
    path.write_bytes(runner.canonical(document))
    with pytest.raises(analysis.LLMAuditAnalysisError, match="attempt_hash"):
        qualification.score_inputs(spec)


def test_cli_writes_immutable_safe_result_and_hash(tmp_path, capsys):
    _, packet, expected = suite()
    run = make_run(tmp_path, packet, expected)
    spec = write_inputs(tmp_path, packet, expected, [run])
    inputs = tmp_path / "inputs.json"
    inputs.write_bytes(runner.canonical(spec))
    output = tmp_path / "qualification.json"
    args = ["--inputs", str(inputs), "--inputs-sha256", analysis.sha(spec),
            "--output", str(output)]
    assert qualification.main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["result_sha256"] == runner.sha(output.read_bytes())
    raw = output.read_text()
    assert '"prompt_text"' not in raw and '"response_text"' not in raw
    assert "stolen password" not in raw
    original = output.read_bytes()
    assert qualification.main(args) == 1
    assert json.loads(capsys.readouterr().out) == {"status": "a161_qualification_rejected"}
    assert output.read_bytes() == original
