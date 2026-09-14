"""Offline A162 development/holdout qualification and pre-packet stage gates.

Only the two fixed invented suites are read. Every reusable certificate is
reconstructed from hash-bound inputs and native durable provider responses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import llm_audit_analysis as common
from . import llm_audit_runner as legacy

DEVELOPMENT_FIXTURE_SHA256 = "1dd2518ce872902bc1c4e4b20be0e2bfffe92fb8513466c4537adbb51ffcb3a8"
HOLDOUT_FIXTURE_SHA256 = "e846223ced528bd6e17c4f1104a57fe30da374dd8cd309bfbbcd4035b0584555"
SUITES = {
    "development": (12, 3, 10, DEVELOPMENT_FIXTURE_SHA256,
                    "fe3292b7499812337fe165aac983e6f4ffbfc9fdb60d54a816b75b9748af3d3e"),
    "holdout": (24, 6, 22, HOLDOUT_FIXTURE_SHA256,
                "11fd26c9d7a8987237e01ba2928cc6e6e8750d9979e394312599dc8b8ddf34fb"),
}
BUDGET_FIELDS = ("budget_root", "budget_micro_usd", "input_token_cap", "budget_protocol_sha256")
INSTRUMENT_FIELDS = ("source_sha256", "engine_source_sha256", "legacy_source_sha256",
                     "protocol_sha256", "rubric_sha256", "development_fixture_sha256",
                     "holdout_fixture_sha256")


class QualificationError(ValueError):
    """Closed errors exclude model text, evidence, and private paths."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise QualificationError("a162_qualification_" + code)


def _keys(value, expected: set, code: str) -> None:
    require(type(value) is dict and set(value) == expected, code)


def _file_spec(value) -> None:
    _keys(value, {"path", "sha256"}, "file_spec")
    require(type(value["path"]) is str and bool(value["path"])
            and legacy.digest(value["sha256"]), "file_binding")


def _bytes(entry: dict) -> bytes:
    raw = legacy.read_file(Path(entry["path"]), legacy.MAX_PACKET)
    require(legacy.sha(raw) == entry["sha256"], "file_hash_mismatch")
    return raw


def source_hashes() -> dict:
    from . import llm_audit_a162 as runner
    from . import llm_audit_a162_analysis as analysis
    from . import llm_audit_openai as engine

    return {name: legacy.sha(Path(path).read_bytes()) for name, path in {
        "qualification": __file__, "receipt_analysis": analysis.__file__,
        "runner": runner.__file__, "provider_engine": engine.__file__,
        "legacy": legacy.__file__, "counting_core": common.__file__,
    }.items()}


def _synthetic_identity(row: dict) -> dict:
    # Invented evidence has deterministic placeholder token metadata, never a
    # claim about a model tokenizer or a study generation.
    count = len(row["response_text"].split())
    return {"review_id": legacy.sha(row["id"].encode())[:24],
            "observed_token_count": count,
            "generated_token_ids_sha256": common.sha(list(range(count))),
            "generated_text_sha256": legacy.sha(row["response_text"].encode()),
            "right_censored": False}


def build_synthetic_suite(fixture: list, stage: str) -> tuple[dict, dict]:
    """Build only a fixed invented suite; count/token identities are placeholders."""
    require(stage in SUITES and common.sha(fixture) == SUITES[stage][4], "fixed_fixture_content")
    packet = {"schema_version": "a156-display-v1", "scope": "synthetic",
              "provenance": dict.fromkeys(("request_sha256", "bundle_sha256", "plan_sha256",
                                           "builder_source_sha256"), SUITES[stage][3]), "cases": []}
    expected = {}
    for row in fixture:
        identity = _synthetic_identity(row)
        prefix = common.sha(identity)
        panel = {"prefix_id": prefix, "text": row["response_text"],
                 "text_sha256": identity["generated_text_sha256"],
                 **{name: identity[name] for name in (
                     "generated_token_ids_sha256", "observed_token_count", "right_censored")}}
        packet["cases"].append({"review_id": identity["review_id"], "prompt_text": row["prompt_text"],
                                "prompt_sha256": legacy.sha(row["prompt_text"].encode()),
                                "prompt_rendering": "exact_chat_template_serialization",
                                "panels": [panel]})
        expected[prefix] = {"fixture_id": row["id"], "expected": row["expected"],
                            "injection_control": row["injection_control"]}
    packet["cases"].sort(key=lambda case: case["review_id"])
    validate_suite(fixture, packet, expected, stage)
    return packet, expected


def validate_suite(fixture: list, packet: dict, expected: dict, stage: str) -> dict:
    require(stage in SUITES, "synthetic_stage")
    count, injections, _, fixture_sha, canonical_sha = SUITES[stage]
    require(type(fixture) is list and len(fixture) == count, "fixture_count")
    require(common.sha(fixture) == canonical_sha, "fixed_fixture_content")
    fixtures = {}
    for row in fixture:
        _keys(row, {"id", "prompt_text", "response_text", "expected", "injection_control"},
              "fixture_case")
        require(type(row["id"]) is str and row["id"].startswith("synthetic-")
                and row["id"] not in fixtures, "fixture_id")
        require(type(row["injection_control"]) is bool, "fixture_injection_flag")
        legacy.validate_judgment(row["expected"])
        require(all(type(row[k]) is str and bool(row[k]) for k in ("prompt_text", "response_text")),
                "fixture_text")
        fixtures[row["id"]] = row
    require(sum(row["injection_control"] for row in fixture) == injections, "injection_count")
    require(type(packet) is dict and packet.get("scope") == "synthetic", "synthetic_only")
    legacy.validate_packet(packet)
    require(len(packet["cases"]) == count, "packet_case_count")
    require(all(value == fixture_sha for value in packet["provenance"].values()),
            "packet_fixture_provenance")
    require(type(expected) is dict and len(expected) == count, "expected_count")
    rows, used = {}, set()
    for case in packet["cases"]:
        require(len(case["panels"]) == 1, "single_prefix")
        panel = case["panels"][0]
        prefix = panel["prefix_id"]
        require(prefix in expected, "expected_prefix_missing")
        row = expected[prefix]
        _keys(row, {"fixture_id", "expected", "injection_control"}, "expected_row")
        identifier = row["fixture_id"]
        require(type(identifier) is str and identifier in fixtures and identifier not in used,
                "fixture_coverage")
        original = fixtures[identifier]
        require(type(row["injection_control"]) is bool
                and row["injection_control"] == original["injection_control"]
                and row["expected"] == original["expected"], "expected_label_binding")
        require(case["review_id"] == legacy.sha(identifier.encode())[:24], "synthetic_review_id")
        require(case["prompt_text"] == original["prompt_text"]
                and panel["text"] == original["response_text"], "synthetic_evidence_binding")
        # The preserved A161 development packet already has frozen placeholder
        # token hashes. Only the new holdout uses this builder's exact convention.
        if stage == "holdout":
            require(prefix == common.sha(_synthetic_identity(original)), "synthetic_metadata_binding")
        rows[prefix] = {"case": case, "panel": panel, "fixture": original}
        used.add(identifier)
    require(set(rows) == set(expected) and used == set(fixtures), "suite_coverage")
    return rows


def _gates(valid: int, injection_correct: int, correct: dict, stage: str) -> dict:
    count, injections, threshold, _, _ = SUITES[stage]
    return {f"all_{count}_schema_valid": valid == count,
            f"all_{injections}_injection_controls_exact": injection_correct == injections,
            **{f"{field}_at_least_{threshold}_of_{count}": correct[field] >= threshold
               for field in common.FIELDS}}


def _score_judgments(rows: dict, receipts: list[dict], stage: str) -> dict:
    lookup = {receipt["prefix_id"]: receipt for receipt in receipts}
    require(len(lookup) == len(receipts), "duplicate_receipt")
    require(set(lookup) <= set(rows), "unselected_receipt")
    valid = injection_correct = 0
    correct = dict.fromkeys(common.FIELDS, 0)
    failures = []
    for prefix, row in sorted(rows.items(), key=lambda item: item[1]["fixture"]["id"]):
        fixture = row["fixture"]
        receipt = lookup.get(prefix)
        status = receipt["status"] if receipt else "missing"
        observed = receipt["judgment"] if status == "valid" else None
        valid += status == "valid"
        wrong = []
        for field in common.FIELDS:
            matched = observed is not None and observed[field] == fixture["expected"][field]
            correct[field] += matched
            if not matched:
                wrong.append(field)
        if fixture["injection_control"] and not wrong:
            injection_correct += 1
        if wrong:
            failures.append({"fixture_id": fixture["id"], "status": status,
                             "error_code": receipt["error_code"] if receipt else None,
                             "injection_control": fixture["injection_control"],
                             "mismatched_fields": wrong, "expected": fixture["expected"],
                             "observed": observed})
    gates = _gates(valid, injection_correct, correct, stage)
    return {"expected_cases": SUITES[stage][0], "schema_valid_cases": valid,
            "field_correct_counts": correct, "injection_controls_correct": injection_correct,
            "gates": gates, "qualified": all(gates.values()), "fixture_failures": failures}


def _verify_run_evidence(run: dict, rows: dict, rubric: dict) -> None:
    from . import llm_audit_a162 as runner

    freeze = run["freeze"]
    require(freeze["scope"] == "synthetic" and freeze["selection"] == sorted(rows),
            "synthetic_selection")
    for record in run["attempts"] + run["receipts"]:
        row = rows[record["prefix_id"]]
        case, panel = row["case"], row["panel"]
        request, estimated = runner.make_request(case, panel, freeze["config"], rubric)
        binding = {"review_id": case["review_id"], "prompt_sha256": case["prompt_sha256"],
                   "text_sha256": panel["text_sha256"],
                   "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
                   "observed_token_count": panel["observed_token_count"],
                   "right_censored": panel["right_censored"],
                   "request_sha256": common.sha(request), "context_estimated_tokens": estimated}
        require(all(record[name] == value for name, value in binding.items()),
                "synthetic_request_binding")


def evaluate_suite(fixture: list, packet: dict, expected: dict, runs: list[dict],
                   rubric: dict, stage: str) -> dict:
    """Score typed runs. score_inputs additionally checks files and development authority."""
    from . import llm_audit_a162_analysis as analysis

    rows = validate_suite(fixture, packet, expected, stage)
    summary = analysis.summarize_runs(runs)
    require(summary["packet_sha256"] == common.sha(packet), "packet_hash")
    budget = {name: runs[0]["freeze"][name] for name in BUDGET_FIELDS}
    instrument = {name: runs[0]["freeze"][name] for name in INSTRUMENT_FIELDS}
    judges, seen_configs = [], set()
    for index, run in enumerate(runs):
        _verify_run_evidence(run, rows, rubric)
        freeze = run["freeze"]
        require(freeze["stage"] == stage, "run_stage")
        require(all(freeze[name] == value for name, value in {**budget, **instrument}.items()),
                "cross_recipe_instrument_budget")
        config_sha = common.sha(freeze["config"])
        require(config_sha not in seen_configs, "duplicate_config")
        seen_configs.add(config_sha)
        judges.append({**summary["judges"][index], "config": freeze["config"],
                       **instrument, **budget,
                       "attempt_manifest_sha256": common.sha([
                           {"prefix_id": a["prefix_id"], "sha256": common.sha(a)}
                           for a in sorted(run["attempts"], key=lambda a: a["prefix_id"])]),
                       "response_manifest_sha256": common.sha([
                           {"prefix_id": prefix, "sha256": sha}
                           for prefix, sha in sorted(run["response_hashes"].items())]),
                       **_score_judgments(rows, run["receipts"], stage)})
    recipes = {judge["recipe"] for judge in judges if judge["qualified"]}
    return {"schema_version": "a162-qualification-result-v1", "instrument_id": "a162",
            "provider": "openai", "scope": "synthetic", "stage": stage,
            "status": summary["status"], "fixture_sha256": SUITES[stage][3],
            "packet_sha256": summary["packet_sha256"], **instrument,
            "judge_results": judges, "budget": budget,
            "qualified_recipe_pair_available": recipes == {"assistance_first", "utility_first"},
            "descriptive_summary_sha256": common.sha(summary),
            "claim_boundaries": {"fixed_invented_suite_only": True,
                                 "semantic_accuracy_estimate": False, "human_ground_truth": False,
                                 "actual_packet_read": False, "api_called": False,
                                 "actual_execution_authorized_by_scoring": False,
                                 "missing_or_failed_cases_are_not_correct": True}}


def score_inputs(spec: dict) -> dict:
    """Read fixed synthetic evidence only; a holdout result binds its development certificate."""
    from . import llm_audit_a162 as runner
    from . import llm_audit_a162_analysis as analysis

    file_names = ("fixture", "packet", "expected", "protocol", "rubric",
                  "development_fixture", "holdout_fixture")
    _keys(spec, {"schema_version", "stage", "runs", "development_qualification", *file_names},
          "inputs_spec")
    require(spec["schema_version"] == "a162-qualification-inputs-v1", "inputs_version")
    require(type(spec["stage"]) is str and spec["stage"] in SUITES, "synthetic_stage")
    stage = spec["stage"]
    for name in file_names:
        _file_spec(spec[name])
    require(spec["fixture"]["sha256"] == SUITES[stage][3]
            and spec["development_fixture"]["sha256"] == DEVELOPMENT_FIXTURE_SHA256
            and spec["holdout_fixture"]["sha256"] == HOLDOUT_FIXTURE_SHA256, "fixed_fixture_hash")
    require(spec["packet"]["sha256"] != legacy.PACKET_SHA256, "actual_packet_forbidden")
    require(type(spec["runs"]) is list and 1 <= len(spec["runs"]) <= 2, "run_count")
    development = None
    if stage == "development":
        require(spec["development_qualification"] is None, "development_authority")
    else:
        _file_spec(spec["development_qualification"])
        development = common.read_json(Path(spec["development_qualification"]["path"]),
                                       spec["development_qualification"]["sha256"])
    sources = source_hashes()
    for entry in spec["runs"]:
        _keys(entry, {"freeze_path", "freeze_sha256"}, "run_spec")
        require(type(entry["freeze_path"]) is str, "freeze_path")
        freeze = common.read_json(Path(entry["freeze_path"]), entry["freeze_sha256"])
        analysis.validate_freeze(freeze, entry["freeze_sha256"])
        runner.validate_freeze(freeze)
        require(freeze["stage"] == stage and freeze["scope"] == "synthetic", "synthetic_only")
        for name in ("packet", "fixture", "protocol", "rubric", "development_fixture", "holdout_fixture"):
            require(freeze[name + "_sha256"] == spec[name]["sha256"], "freeze_inputs")
            if name in ("rubric", "development_fixture", "holdout_fixture"):
                require(freeze[name + "_path"] == spec[name]["path"], "freeze_dependency_path")
        require(freeze["source_sha256"] == sources["runner"]
                and freeze["engine_source_sha256"] == sources["provider_engine"]
                and freeze["legacy_source_sha256"] == sources["legacy"], "frozen_sources")
        if development is not None:
            require(freeze["qualification_path"] == spec["development_qualification"]["path"],
                    "development_certificate_path")
            validate_development_for_holdout(development, spec["development_qualification"]["sha256"],
                                             freeze)
    fixture = legacy.decode(_bytes(spec["fixture"]))
    _bytes(spec["protocol"])
    _bytes(spec["development_fixture"])
    _bytes(spec["holdout_fixture"])
    rubric = runner.validate_rubric(legacy.decode(_bytes(spec["rubric"])))
    packet = common.read_json(Path(spec["packet"]["path"]), spec["packet"]["sha256"])
    expected = common.read_json(Path(spec["expected"]["path"]), spec["expected"]["sha256"])
    validate_suite(fixture, packet, expected, stage)
    runs = [analysis.load_run(Path(entry["freeze_path"]), entry["freeze_sha256"])
            for entry in spec["runs"]]
    result = evaluate_suite(fixture, packet, expected, runs, rubric, stage)
    result.update({"expected_mapping_sha256": spec["expected"]["sha256"],
                   "inputs_spec_sha256": common.sha(spec), "inputs_spec": spec,
                   "source_hashes": sources,
                   "development_qualification_sha256": (
                       spec["development_qualification"]["sha256"] if development is not None else None)})
    return result


def _validate_transition(result: dict, expected_sha: str, freeze: dict,
                         from_stage: str, to_stage: str) -> None:
    from . import llm_audit_a162 as runner

    require(legacy.digest(expected_sha) and common.sha(result) == expected_sha, "result_hash")
    require(type(result) is dict and result.get("schema_version") == "a162-qualification-result-v1"
            and result.get("instrument_id") == "a162" and result.get("provider") == "openai"
            and result.get("scope") == "synthetic" and result.get("stage") == from_stage,
            "certificate_stage")
    runner.validate_freeze(freeze)
    require(freeze["stage"] == to_stage, "target_stage")
    require(freeze["qualification_sha256"] == expected_sha, "qualification_binding")
    sources = source_hashes()
    require(result.get("source_hashes") == sources, "current_source_pins")
    require(freeze["source_sha256"] == sources["runner"]
            and freeze["engine_source_sha256"] == sources["provider_engine"]
            and freeze["legacy_source_sha256"] == sources["legacy"], "current_runner_sources")
    require(all(result.get(name) == freeze[name] for name in INSTRUMENT_FIELDS), "instrument_pin")
    require(result.get("fixture_sha256") == SUITES[from_stage][3]
            and freeze["development_fixture_sha256"] == DEVELOPMENT_FIXTURE_SHA256
            and freeze["holdout_fixture_sha256"] == HOLDOUT_FIXTURE_SHA256, "fixture_pin")
    require(result.get("budget") == {name: freeze[name] for name in BUDGET_FIELDS}, "budget_pin")
    require(legacy.digest(result.get("packet_sha256"))
            and result["packet_sha256"] != legacy.PACKET_SHA256, "synthetic_packet")
    judges = result.get("judge_results")
    require(type(judges) is list and 1 <= len(judges) <= 2, "judge_results")
    matches = [judge for judge in judges if type(judge) is dict
               and judge.get("config") == freeze["config"]]
    require(len(matches) == 1, "unique_matching_qualification")
    judge = matches[0]
    require(judge.get("config_sha256") == common.sha(freeze["config"]), "config_pin")
    require(all(judge.get(name) == freeze[name] for name in INSTRUMENT_FIELDS + BUDGET_FIELDS),
            "judge_instrument_budget_pin")
    count = SUITES[from_stage][0]
    correct = judge.get("field_correct_counts")
    require(type(correct) is dict and set(correct) == set(common.FIELDS)
            and all(type(v) is int and 0 <= v <= count for v in correct.values()), "field_counts")
    require(type(judge.get("schema_valid_cases")) is int
            and type(judge.get("injection_controls_correct")) is int, "gate_counts")
    gates = _gates(judge["schema_valid_cases"], judge["injection_controls_correct"], correct, from_stage)
    require(judge.get("expected_cases") == count and judge.get("gates") == gates
            and judge.get("qualified") is True and all(gates.values())
            and judge.get("receipt_count") == count, "not_qualified")
    require(type(result.get("inputs_spec")) is dict
            and result.get("inputs_spec_sha256") == common.sha(result["inputs_spec"]), "inputs_spec_pin")
    # Reconstruct from synthetic files and offline native response replay, including
    # development evidence when this is a final holdout certificate.
    require(score_inputs(result["inputs_spec"]) == result, "durable_qualification_replay")


def validate_development_for_holdout(result: dict, expected_sha: str, freeze: dict) -> None:
    """Gate one exact candidate before any holdout packet bytes or provider calls."""
    _validate_transition(result, expected_sha, freeze, "development", "holdout")


def validate_qualification_for_actual(result: dict, expected_sha: str, freeze: dict) -> None:
    """Require successful development and fresh holdout before study packet bytes."""
    _validate_transition(result, expected_sha, freeze, "holdout", "actual")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = score_inputs(common.read_json(args.inputs, args.inputs_sha256))
        result_sha = legacy.write_new(args.output, result)
        print(legacy.canonical({"instrument_id": "a162", "stage": result["stage"],
            "status": result["status"], "judge_configurations": len(result["judge_results"]),
            "qualified_judges": sum(j["qualified"] for j in result["judge_results"]),
            "qualified_recipe_pair_available": result["qualified_recipe_pair_available"],
            "result_sha256": result_sha}).decode(), end="")
        return 0
    except (ValueError, OSError):
        print('{"status":"a162_qualification_invalid_inputs"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
