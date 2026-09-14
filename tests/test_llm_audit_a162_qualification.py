from __future__ import annotations

from copy import deepcopy
import fcntl
import json
from pathlib import Path

import pytest

from lexical_prompt_study import llm_audit_a162 as runner
from lexical_prompt_study import llm_audit_a162_qualification as q
from lexical_prompt_study import llm_audit_analysis as common
from lexical_prompt_study import llm_audit_openai as engine
from lexical_prompt_study import llm_audit_runner as legacy
from test_llm_audit_openai_qualification import synthetic_response

PLANS = Path(__file__).parents[1] / "plans"


def file_spec(path):
    return {"path": str(path), "sha256": legacy.sha(path.read_bytes())}


def write(path, value):
    path.write_bytes(legacy.canonical(value))
    return file_spec(path)


def fixture(stage):
    path = PLANS / ("llm_audit_qualification_a161.json" if stage == "development"
                    else "llm_audit_holdout_a162.json")
    return json.loads(path.read_bytes())


def gate_inputs(stage):
    rows = {f"{index:064x}": {"fixture": row} for index, row in enumerate(fixture(stage))}
    receipts = [{"prefix_id": prefix, "status": "valid", "error_code": "none",
                 "judgment": deepcopy(row["fixture"]["expected"])} for prefix, row in rows.items()]
    return rows, receipts


def make_spec(tmp_path, stage, recipes=("assistance_first", "utility_first"),
              development=None, mutate=None):
    root = tmp_path / stage
    root.mkdir()
    packet, expected = q.build_synthetic_suite(fixture(stage), stage)
    rubric = json.loads((PLANS / "llm_audit_rubric_a162.json").read_bytes())
    spec = {"schema_version": "a162-qualification-inputs-v1", "stage": stage,
            "development_fixture": file_spec(PLANS / "llm_audit_qualification_a161.json"),
            "holdout_fixture": file_spec(PLANS / "llm_audit_holdout_a162.json"),
            "rubric": file_spec(PLANS / "llm_audit_rubric_a162.json"),
            "protocol": file_spec(PLANS / "llm_label_audit_a162.md"),
            "packet": write(root / "packet.synthetic.json", packet),
            "expected": write(root / "expected.synthetic.json", expected),
            "runs": [], "development_qualification": development}
    spec["fixture"] = spec[stage + "_fixture"]
    for index, recipe in enumerate(recipes):
        config = {"provider": "openai", "model_name": engine.MODEL, "endpoint": engine.ENDPOINT,
                  "recipe": recipe, "temperature": 0, "max_output_tokens": 192,
                  "input_token_limit": 32768, "input_micro_usd_per_token": 2,
                  "output_micro_usd_per_token": 8, "instrument_id": "a162",
                  "rubric_sha256": spec["rubric"]["sha256"]}
        output = root / f"run-{index}"
        output.mkdir()
        freeze = {"schema_version": "a162-freeze-v1", "instrument_id": "a162", "stage": stage,
                  "scope": "synthetic", "run_id": f"a162-{stage}-{index}",
                  "source_sha256": legacy.sha(Path(runner.__file__).read_bytes()),
                  "engine_source_sha256": runner.ENGINE_SHA256,
                  "legacy_source_sha256": runner.LEGACY_SHA256,
                  "config": config, "selection": sorted(expected), "output_root": str(output),
                  "budget_root": runner.BUDGET_ROOT, "budget_micro_usd": 10_000_000,
                  "input_token_cap": 5_000_000, "budget_protocol_sha256": runner.BUDGET_PROTOCOL_SHA256,
                  "timeout_seconds": 60, "max_seconds": 1200, "concurrency": 1,
                  "execution_authorized": True,
                  "qualification_sha256": development["sha256"] if development else None,
                  "qualification_path": development["path"] if development else None}
        for name in ("packet", "fixture", "protocol", "rubric", "development_fixture", "holdout_fixture"):
            freeze[name + "_sha256"] = spec[name]["sha256"]
            if name in ("rubric", "development_fixture", "holdout_fixture"):
                freeze[name + "_path"] = spec[name]["path"]
        freeze_entry = write(root / f"freeze-{index}.json", freeze)
        freeze_sha = freeze_entry["sha256"]
        spec["runs"].append({"freeze_path": freeze_entry["path"], "freeze_sha256": freeze_sha})
        (output / ".writer.lock").touch()
        write(output / "manifest.json", {"schema_version": "a162-store-v1", "instrument_id": "a162",
              "freeze_sha256": freeze_sha, "config_sha256": common.sha(config),
              "selected_prefixes": len(expected)})
        for kind in ("attempts", "responses", "receipts"):
            (output / kind).mkdir()
        for case in packet["cases"]:
            panel = case["panels"][0]
            prefix = panel["prefix_id"]
            request, estimate = runner.make_request(case, panel, config, rubric)
            bindings = engine._bindings(freeze, freeze_sha, common.sha(config), case, panel,
                                         request, estimate)
            attempt = {"schema_version": "a161-openai-attempt-v1", **bindings,
                       "budget_entry_id": common.sha({"freeze_sha256": freeze_sha, "prefix_id": prefix}),
                       "reserved_micro_usd": estimate * 2 + 192 * 8}
            attempt_entry = write(output / "attempts" / (prefix + ".json"), attempt)
            judgment = deepcopy(expected[prefix]["expected"])
            if mutate is not None:
                mutate(expected[prefix], judgment, recipe)
            response_path = output / "responses" / (prefix + ".json")
            response_path.write_bytes(synthetic_response(judgment))
            receipt = engine._receipt(attempt, attempt_entry["sha256"], response_path, 1.0)
            write(output / "receipts" / (prefix + ".json"), receipt)
    return spec


def freeze_from(spec, index=0):
    return json.loads(Path(spec["runs"][index]["freeze_path"]).read_bytes())


def actual_freeze(spec, result, tmp_path, index=0):
    freeze = freeze_from(spec, index)
    freeze.update(stage="actual", scope="verified_a156", run_id="a162-actual-test",
                  packet_sha256=legacy.PACKET_SHA256, selection=[f"{i:064x}" for i in range(287)],
                  output_root=str(tmp_path / "actual-output"), qualification_sha256=common.sha(result),
                  qualification_path=str(tmp_path / "final.qualification.json"))
    return freeze


def stages(tmp_path, recipes=("assistance_first", "utility_first"), mutate=None):
    development = make_spec(tmp_path, "development", recipes)
    development_result = q.score_inputs(development)
    certificate = write(tmp_path / "development.result.json", development_result)
    holdout = make_spec(tmp_path, "holdout", recipes, certificate, mutate)
    return development, holdout


@pytest.mark.parametrize("stage,count,injections", [("development", 12, 3), ("holdout", 24, 6)])
def test_exact_stage_gates_and_frozen_fixture(stage, count, injections):
    rows, receipts = gate_inputs(stage)
    result = q._score_judgments(rows, receipts, stage)
    assert result["qualified"] and result["expected_cases"] == count
    assert result["field_correct_counts"] == dict.fromkeys(common.FIELDS, count)
    assert result["injection_controls_correct"] == injections
    packet, expected = q.build_synthetic_suite(fixture(stage), stage)
    assert len(q.validate_suite(fixture(stage), packet, expected, stage)) == count


@pytest.mark.parametrize("field", list(common.FIELDS))
@pytest.mark.parametrize("stage", ["development", "holdout"])
def test_every_field_must_meet_own_threshold(stage, field):
    rows, receipts = gate_inputs(stage)
    for receipt in receipts[:3]:
        receipt["judgment"][field] = "uncertain"
    score = q._score_judgments(rows, receipts, stage)
    assert not score["qualified"]
    assert score["field_correct_counts"][field] == q.SUITES[stage][0] - 3


@pytest.mark.parametrize("field", list(common.FIELDS))
def test_single_holdout_injection_failure_blocks_even_above_field_threshold(field):
    rows, receipts = gate_inputs("holdout")
    receipts[-1]["judgment"][field] = "uncertain"
    score = q._score_judgments(rows, receipts, "holdout")
    assert not score["qualified"] and score["injection_controls_correct"] == 5
    assert score["field_correct_counts"][field] == 23


def test_holdout_schema_failure_and_missing_are_not_correct():
    rows, receipts = gate_inputs("holdout")
    receipts.pop()
    receipts[0].update(status="schema_failed", error_code="invalid_judgment", judgment=None)
    score = q._score_judgments(rows, receipts, "holdout")
    assert not score["qualified"] and score["schema_valid_cases"] == 22
    assert {failure["status"] for failure in score["fixture_failures"]} == {"missing", "schema_failed"}


def test_two_stage_durable_replay_qualifies_exact_configs_without_api(tmp_path, monkeypatch):
    monkeypatch.setattr(engine.OpenAIClient, "request", lambda *a, **k: pytest.fail("API called"))
    _, holdout = stages(tmp_path)
    result = q.score_inputs(holdout)
    assert result["qualified_recipe_pair_available"]
    assert result["stage"] == "holdout" and result["instrument_id"] == "a162"
    assert result["development_qualification_sha256"] == holdout["development_qualification"]["sha256"]
    assert all(j["field_correct_counts"] == dict.fromkeys(common.FIELDS, 24)
               for j in result["judge_results"])
    for index in range(2):
        q.validate_qualification_for_actual(result, common.sha(result),
                                            actual_freeze(holdout, result, tmp_path, index))
    assert '"prompt_text"' not in json.dumps(result)
    assert '"observed_response_prefix"' not in json.dumps(result)


def test_one_qualified_recipe_is_explicitly_partial(tmp_path):
    _, holdout = stages(tmp_path, ("assistance_first",))
    result = q.score_inputs(holdout)
    assert len(result["judge_results"]) == 1 and result["judge_results"][0]["qualified"]
    assert not result["qualified_recipe_pair_available"]
    q.validate_qualification_for_actual(result, common.sha(result), actual_freeze(holdout, result, tmp_path))
    actual = actual_freeze(holdout, result, tmp_path)
    actual["config"]["recipe"] = "utility_first"
    with pytest.raises(q.QualificationError, match="unique_matching"):
        q.validate_qualification_for_actual(result, common.sha(result), actual)


def test_failed_development_stops_before_holdout_packet_read(tmp_path, monkeypatch):
    def wrong(row, judgment, recipe):
        if row["injection_control"]:
            judgment["safe_task_completion"] = "uncertain"
    development = make_spec(tmp_path, "development", mutate=wrong)
    failed = q.score_inputs(development)
    certificate = write(tmp_path / "development.result.json", failed)
    holdout = make_spec(tmp_path, "holdout", development=certificate)
    packet_path = Path(holdout["packet"]["path"])
    read = common.read_json
    def checked(path, sha):
        assert path != packet_path, "holdout packet was read"
        return read(path, sha)
    monkeypatch.setattr(common, "read_json", checked)
    with pytest.raises(q.QualificationError, match="not_qualified"):
        q.score_inputs(holdout)


def test_failed_holdout_cannot_authorize_actual(tmp_path):
    def wrong(row, judgment, recipe):
        if row["injection_control"]:
            judgment["safe_task_completion"] = "uncertain"
    _, holdout = stages(tmp_path, mutate=wrong)
    result = q.score_inputs(holdout)
    assert not result["qualified_recipe_pair_available"]
    with pytest.raises(q.QualificationError, match="not_qualified"):
        q.validate_qualification_for_actual(result, common.sha(result), actual_freeze(holdout, result, tmp_path))


@pytest.mark.parametrize("mutation", [
    lambda f: f["config"].update(input_token_limit=65536),
    lambda f: f["config"].update(model_name="gpt-4.1"),
    lambda f: f.update(rubric_sha256="f" * 64),
    lambda f: f.update(holdout_fixture_sha256="f" * 64),
    lambda f: f.update(source_sha256="f" * 64),
    lambda f: f.update(engine_source_sha256="f" * 64),
    lambda f: f.update(budget_root=f["budget_root"] + "-new"),
    lambda f: f.update(budget_protocol_sha256="f" * 64),
    lambda f: f.update(input_token_cap=4_000_000),
    lambda f: f.update(qualification_sha256="f" * 64),
])
def test_exact_config_instrument_and_global_budget_reuse(tmp_path, mutation):
    _, holdout = stages(tmp_path, ("assistance_first",))
    result = q.score_inputs(holdout)
    actual = actual_freeze(holdout, result, tmp_path)
    mutation(actual)
    with pytest.raises(ValueError):
        q.validate_qualification_for_actual(result, common.sha(result), actual)


@pytest.mark.parametrize("mutation", [
    lambda r: r["source_hashes"].update(qualification="f" * 64),
    lambda r: r["source_hashes"].update(receipt_analysis="f" * 64),
    lambda r: r["source_hashes"].update(counting_core="f" * 64),
    lambda r: r["judge_results"][0].update(response_manifest_sha256="f" * 64),
    lambda r: r["judge_results"][0]["field_correct_counts"].update(refusal_present=21),
    lambda r: r.update(development_qualification_sha256="f" * 64),
])
def test_rehashed_forged_certificate_cannot_bypass_replay(tmp_path, mutation):
    _, holdout = stages(tmp_path, ("assistance_first",))
    result = q.score_inputs(holdout)
    mutation(result)
    with pytest.raises(ValueError):
        q.validate_qualification_for_actual(result, common.sha(result), actual_freeze(holdout, result, tmp_path))


@pytest.mark.parametrize("changed_stage", ["development", "holdout"])
def test_durable_native_body_tampering_invalidates_final_certificate(tmp_path, changed_stage):
    development, holdout = stages(tmp_path, ("assistance_first",))
    result = q.score_inputs(holdout)
    target = freeze_from(development if changed_stage == "development" else holdout)
    response = Path(target["output_root"]) / "responses" / (target["selection"][0] + ".json")
    response.write_bytes(b"{}\n")
    with pytest.raises(ValueError):
        q.validate_qualification_for_actual(result, common.sha(result), actual_freeze(holdout, result, tmp_path))


def test_development_certificate_cannot_be_used_directly_for_actual(tmp_path):
    spec = make_spec(tmp_path, "development", ("assistance_first",))
    result = q.score_inputs(spec)
    with pytest.raises(q.QualificationError, match="certificate_stage"):
        q.validate_qualification_for_actual(result, common.sha(result), actual_freeze(spec, result, tmp_path))


def test_actual_packet_hash_forbidden_before_any_read(tmp_path, monkeypatch):
    spec = make_spec(tmp_path, "development", ("assistance_first",))
    spec["packet"] = {"path": "DO_NOT_READ", "sha256": legacy.PACKET_SHA256}
    monkeypatch.setattr(common, "read_json", lambda *a: pytest.fail("actual packet read"))
    with pytest.raises(q.QualificationError, match="actual_packet_forbidden"):
        q.score_inputs(spec)


def test_active_writer_prevents_certificate(tmp_path):
    spec = make_spec(tmp_path, "development", ("assistance_first",))
    with (Path(freeze_from(spec)["output_root"]) / ".writer.lock").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((ValueError, OSError)):
            q.score_inputs(spec)


def test_synthetic_label_or_evidence_substitution_rejected():
    rows = fixture("holdout")
    packet, expected = q.build_synthetic_suite(rows, "holdout")
    prefix = next(iter(expected))
    expected[prefix]["expected"] = dict.fromkeys(common.FIELDS, "uncertain")
    with pytest.raises(q.QualificationError, match="expected_label_binding"):
        q.validate_suite(rows, packet, expected, "holdout")
    rows[0]["response_text"] += " altered"
    with pytest.raises(q.QualificationError, match="fixed_fixture_content"):
        q.build_synthetic_suite(rows, "holdout")


def test_runner_checks_final_certificate_before_actual_packet_bytes(tmp_path, monkeypatch):
    _, holdout = stages(tmp_path, ("assistance_first",))
    result = q.score_inputs(holdout)
    actual = actual_freeze(holdout, result, tmp_path)
    actual["config"]["input_token_limit"] = 65536
    write(Path(actual["qualification_path"]), result)
    freeze_spec = write(tmp_path / "actual.freeze.json", actual)
    forbidden = tmp_path / "study-do-not-read.json"
    read = runner.read_file
    def checked(path, limit):
        assert path != forbidden, "actual packet was read"
        return read(path, limit)
    monkeypatch.setattr(runner, "read_file", checked)
    with pytest.raises(q.QualificationError, match="unique_matching"):
        runner._load_inputs(forbidden, Path(freeze_spec["path"]), freeze_spec["sha256"],
                            Path(holdout["protocol"]["path"]), Path(actual["output_root"]))
