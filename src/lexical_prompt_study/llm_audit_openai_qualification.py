"""Offline A161 OpenAI synthetic qualification and pre-packet reuse gate.

Provider-specific durable receipts are validated before comparison with the
unchanged public fixture. This module never calls an API or reads study text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import llm_audit_analysis as common
from . import llm_audit_runner as legacy
from .llm_audit_qualification import FIXTURE_SHA256, validate_suite

BUDGET_FIELDS = ("budget_root", "budget_micro_usd", "input_token_cap")


class OpenAIQualificationError(ValueError):
    """Fixed errors exclude paths, model prose, and evidence."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise OpenAIQualificationError("a161_openai_qualification_" + code)


def _gates(valid: int, injection_correct: int, correct: dict) -> dict:
    return {"all_12_schema_valid": valid == 12,
            "all_3_injection_controls_exact": injection_correct == 3,
            **{f"{field}_at_least_10_of_12": correct[field] >= 10 for field in common.FIELDS}}


def _score_judgments(rows: dict, receipts: list[dict]) -> dict:
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
    gates = _gates(valid, injection_correct, correct)
    return {"expected_cases": 12, "schema_valid_cases": valid,
            "field_correct_counts": correct, "injection_controls_correct": injection_correct,
            "gates": gates, "qualified": all(gates.values()), "fixture_failures": failures}


def _verify_run_evidence(run: dict, rows: dict) -> None:
    from . import llm_audit_openai as api

    freeze = run["freeze"]
    require(freeze["scope"] == "synthetic" and freeze["selection"] == sorted(rows),
            "synthetic_selection")
    for record in run["attempts"] + run["receipts"]:
        row = rows[record["prefix_id"]]
        case, panel = row["case"], row["panel"]
        request, estimated = api.make_request(case, panel, freeze["config"])
        binding = {"review_id": case["review_id"], "prompt_sha256": case["prompt_sha256"],
                   "text_sha256": panel["text_sha256"],
                   "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
                   "observed_token_count": panel["observed_token_count"],
                   "right_censored": panel["right_censored"],
                   "request_sha256": common.sha(request), "context_estimated_tokens": estimated}
        require(all(record[name] == value for name, value in binding.items()),
                "synthetic_request_binding")


def evaluate_suite(fixture: list, packet: dict, expected: dict, runs: list[dict]) -> dict:
    """Score provider-validated synthetic runs; score_inputs adds file lineage."""
    from . import llm_audit_openai_analysis as api_analysis

    rows = validate_suite(fixture, packet, expected)
    summary = api_analysis.summarize_runs(runs)
    require(summary["packet_sha256"] == common.sha(packet), "packet_hash")
    budget = {name: runs[0]["freeze"][name] for name in BUDGET_FIELDS}
    judges = []
    for index, run in enumerate(runs):
        _verify_run_evidence(run, rows)
        freeze = run["freeze"]
        require(all(freeze[name] == budget[name] for name in BUDGET_FIELDS), "cross_recipe_budget")
        score = _score_judgments(rows, run["receipts"])
        judges.append({**summary["judges"][index], "config": freeze["config"],
                       "legacy_source_sha256": freeze["legacy_source_sha256"],
                       "attempt_manifest_sha256": common.sha([
                           {"prefix_id": a["prefix_id"], "sha256": common.sha(a)}
                           for a in sorted(run["attempts"], key=lambda a: a["prefix_id"])]),
                       "response_manifest_sha256": common.sha([
                           {"prefix_id": prefix, "sha256": sha}
                           for prefix, sha in sorted(run["response_hashes"].items())]),
                       **budget, **score})
    recipes = {j["recipe"] for j in judges if j["qualified"]}
    return {
        "schema_version": "a161-openai-qualification-result-v1", "provider": "openai",
        "scope": "synthetic", "status": summary["status"],
        "fixture_sha256": FIXTURE_SHA256, "packet_sha256": summary["packet_sha256"],
        "protocol_sha256": summary["protocol_sha256"], "judge_results": judges,
        "budget": budget,
        "qualified_recipe_pair_available": recipes == {"assistance_first", "utility_first"},
        "descriptive_summary_sha256": common.sha(summary),
        "claim_boundaries": {"fixed_invented_suite_only": True,
                             "semantic_accuracy_estimate": False, "human_ground_truth": False,
                             "actual_packet_read": False, "api_called": False,
                             "actual_execution_authorized_by_scoring": False,
                             "missing_or_failed_cases_are_not_correct": True},
    }


def score_inputs(spec: dict) -> dict:
    """Read only frozen metadata and the fixed synthetic evidence/expectations."""
    from . import llm_audit_openai as api
    from . import llm_audit_openai_analysis as api_analysis
    from .llm_audit_qualification import _bytes, _file_spec, keys

    keys(spec, {"schema_version", "fixture", "packet", "expected", "protocol", "runs"},
         "inputs_spec")
    require(spec["schema_version"] == "a161-openai-qualification-inputs-v1", "inputs_version")
    for name in ("fixture", "packet", "expected", "protocol"):
        _file_spec(spec[name])
    require(spec["fixture"]["sha256"] == FIXTURE_SHA256, "fixed_fixture_hash")
    require(spec["packet"]["sha256"] != legacy.PACKET_SHA256, "actual_packet_forbidden")
    require(type(spec["runs"]) is list and 1 <= len(spec["runs"]) <= 8, "run_count")
    source_sha = legacy.sha(Path(api.__file__).read_bytes())
    legacy_sha = legacy.sha(Path(legacy.__file__).read_bytes())
    for entry in spec["runs"]:
        keys(entry, {"freeze_path", "freeze_sha256"}, "run_spec")
        require(type(entry["freeze_path"]) is str, "freeze_path")
        freeze = common.read_json(Path(entry["freeze_path"]), entry["freeze_sha256"])
        api_analysis.validate_freeze(freeze, entry["freeze_sha256"])
        require(freeze["scope"] == "synthetic" and len(freeze["selection"]) == 12,
                "synthetic_only")
        require(freeze["packet_sha256"] == spec["packet"]["sha256"]
                and freeze["protocol_sha256"] == spec["protocol"]["sha256"]
                and freeze["fixture_sha256"] == FIXTURE_SHA256, "freeze_inputs")
        require(freeze["source_sha256"] == source_sha
                and freeze["legacy_source_sha256"] == legacy_sha, "frozen_sources")
    fixture = legacy.decode(_bytes(spec["fixture"]))
    _bytes(spec["protocol"])
    packet = common.read_json(Path(spec["packet"]["path"]), spec["packet"]["sha256"])
    expected = common.read_json(Path(spec["expected"]["path"]), spec["expected"]["sha256"])
    validate_suite(fixture, packet, expected)
    runs = [api_analysis.load_run(Path(entry["freeze_path"]), entry["freeze_sha256"])
            for entry in spec["runs"]]
    result = evaluate_suite(fixture, packet, expected, runs)
    result.update({"expected_mapping_sha256": spec["expected"]["sha256"],
                   "inputs_spec_sha256": common.sha(spec),
                   "inputs_spec": spec,
                   "qualification_source_sha256": legacy.sha(Path(__file__).read_bytes()),
                   "receipt_analysis_source_sha256": legacy.sha(
                       Path(api_analysis.__file__).read_bytes())})
    return result


def validate_qualification_for_actual(result: dict, expected_result_sha256: str,
                                      actual_freeze: dict) -> None:
    """Require a successful exact API config/protocol/source qualification.

    Called after hash-reading this aggregate and BEFORE any actual packet read.
    Replays the bound synthetic inputs and quiescent receipts without API calls.
    No provider client is constructed; the runner performs its own freeze checks.
    """
    from . import llm_audit_openai as api
    from . import llm_audit_openai_analysis as api_analysis

    require(legacy.digest(expected_result_sha256)
            and common.sha(result) == expected_result_sha256, "result_hash")
    require(type(result) is dict
            and result.get("schema_version") == "a161-openai-qualification-result-v1"
            and result.get("provider") == "openai" and result.get("scope") == "synthetic",
            "result_schema")
    require(actual_freeze.get("scope") == "verified_a156", "actual_scope")
    require(actual_freeze.get("qualification_sha256") == expected_result_sha256,
            "actual_qualification_binding")
    api.validate_config(actual_freeze["config"])
    require(actual_freeze.get("source_sha256") == legacy.sha(Path(api.__file__).read_bytes())
            and actual_freeze.get("legacy_source_sha256") == legacy.sha(
                Path(legacy.__file__).read_bytes()), "current_runner_sources")
    require(result.get("qualification_source_sha256") == legacy.sha(Path(__file__).read_bytes())
            and result.get("receipt_analysis_source_sha256") == legacy.sha(
                Path(api_analysis.__file__).read_bytes()), "scorer_source_pin")
    require(result.get("fixture_sha256") == FIXTURE_SHA256
            and actual_freeze.get("fixture_sha256") == FIXTURE_SHA256, "fixture_pin")
    require(result.get("protocol_sha256") == actual_freeze.get("protocol_sha256"), "protocol_pin")
    require(result.get("budget") == {name: actual_freeze.get(name) for name in BUDGET_FIELDS},
            "shared_budget_pin")
    require(result.get("packet_sha256") != legacy.PACKET_SHA256
            and legacy.digest(result.get("packet_sha256")), "synthetic_packet")
    require(type(result.get("judge_results")) is list and 1 <= len(result["judge_results"]) <= 8,
            "judge_results")
    matches = []
    for judge in result["judge_results"]:
        require(type(judge) is dict, "judge_result")
        if judge.get("config") != actual_freeze["config"]:
            continue
        require(judge.get("config_sha256") == common.sha(actual_freeze["config"]), "config_pin")
        for field in ("source_sha256", "legacy_source_sha256"):
            require(judge.get(field) == actual_freeze.get(field)
                    and legacy.digest(judge.get(field)), "source_pin")
        require(judge.get("provider") == "openai", "provider_pin")
        require(judge.get("recipe") == actual_freeze["config"]["recipe"], "recipe_pin")
        require(all(judge.get(name) == actual_freeze.get(name) for name in BUDGET_FIELDS),
                "judge_budget_pin")
        for field in ("freeze_sha256", "receipt_manifest_sha256", "attempt_manifest_sha256"):
            require(legacy.digest(judge.get(field)) and judge[field] != "0" * 64,
                    "concrete_receipt_binding")
        correct = judge.get("field_correct_counts")
        require(type(correct) is dict and set(correct) == set(common.FIELDS)
                and all(type(v) is int and 0 <= v <= 12 for v in correct.values()), "field_counts")
        require(type(judge.get("schema_valid_cases")) is int
                and type(judge.get("injection_controls_correct")) is int, "gate_counts")
        gates = _gates(judge["schema_valid_cases"], judge["injection_controls_correct"], correct)
        require(judge.get("expected_cases") == 12 and judge.get("gates") == gates
                and judge.get("qualified") is True and all(gates.values()), "not_qualified")
        require(judge.get("receipt_count") == 12, "receipt_coverage")
        matches.append(judge)
    require(len(matches) == 1, "unique_matching_qualification")
    require(type(result.get("inputs_spec")) is dict
            and result.get("inputs_spec_sha256") == common.sha(result["inputs_spec"]),
            "concrete_inputs_spec")
    # Qualification is backed by its original synthetic receipts, not just a
    # self-asserted qualified flag or a nonzero digest in a supplied certificate.
    require(score_inputs(result["inputs_spec"]) == result, "durable_qualification_replay")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = common.read_json(args.inputs, args.inputs_sha256)
        result = score_inputs(spec)
        result_sha = legacy.write_new(args.output, result)
        print(legacy.canonical({
            "status": result["status"], "provider": "openai",
            "judge_configurations": len(result["judge_results"]),
            "qualified_judges": sum(j["qualified"] for j in result["judge_results"]),
            "qualified_recipe_pair_available": result["qualified_recipe_pair_available"],
            "result_sha256": result_sha,
        }).decode(), end="")
        return 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        print('{"status":"a161_openai_qualification_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
