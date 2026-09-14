"""Score the fixed A161 synthetic suite from quiescent durable judge receipts.

No model calls, actual-packet access, or corpus scoring. CLI outputs only counts
and hashes; the immutable result includes failures by public synthetic-case ID.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import llm_audit_analysis as analysis
from . import llm_audit_runner as runner


FIXTURE_SHA256 = "1dd2518ce872902bc1c4e4b20be0e2bfffe92fb8513466c4537adbb51ffcb3a8"
FIXTURE_CANONICAL_SHA256 = "fe3292b7499812337fe165aac983e6f4ffbfc9fdb60d54a816b75b9748af3d3e"
MODELS = {"gemma3:4b", "mistral:latest", "gemma3:27b-it-qat"}
PIN_FIELDS = (
    "model_name", "model_digest", "ollama_version", "recipe", "seed", "num_ctx",
    "num_predict", "num_thread", "num_gpu",
)


class QualificationError(ValueError):
    """Closed errors never echo contents, model output, or filesystem paths."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise QualificationError("a161_qualification_" + code)


def keys(value: Any, expected: set[str], code: str) -> None:
    require(type(value) is dict and set(value) == expected, code)


def _file_spec(value: Any) -> None:
    keys(value, {"path", "sha256"}, "file_spec")
    require(type(value["path"]) is str and bool(value["path"]), "file_path")
    require(runner.digest(value["sha256"]), "file_hash")


def _bytes(entry: dict) -> bytes:
    raw = runner.read_file(Path(entry["path"]), runner.MAX_PACKET)
    require(runner.sha(raw) == entry["sha256"], "file_hash_mismatch")
    return raw


def validate_suite(fixture: list, packet: dict, expected: dict) -> dict[str, dict]:
    """Bind all twelve expected answers to the exact invented packet evidence."""
    require(type(fixture) is list and len(fixture) == 12, "fixture_count")
    require(analysis.sha(fixture) == FIXTURE_CANONICAL_SHA256, "fixed_fixture_content")
    fixtures = {}
    for case in fixture:
        keys(case, {"id", "prompt_text", "response_text", "expected", "injection_control"},
             "fixture_case")
        require(type(case["id"]) is str and case["id"].startswith("synthetic-"), "fixture_id")
        require(case["id"] not in fixtures, "fixture_duplicate")
        require(type(case["injection_control"]) is bool, "fixture_injection_flag")
        runner.validate_judgment(case["expected"])
        require(all(type(case[k]) is str and bool(case[k])
                    for k in ("prompt_text", "response_text")), "fixture_text")
        fixtures[case["id"]] = case
    require(sum(c["injection_control"] for c in fixture) == 3, "injection_count")
    require(type(packet) is dict and packet.get("scope") == "synthetic", "synthetic_only")
    runner.validate_packet(packet)
    require(len(packet["cases"]) == 12, "packet_case_count")
    require(all(value == FIXTURE_SHA256 for value in packet["provenance"].values()),
            "packet_fixture_provenance")
    require(type(expected) is dict and len(expected) == 12, "expected_count")
    rows, used = {}, set()
    for case in packet["cases"]:
        require(len(case["panels"]) == 1, "synthetic_single_prefix")
        panel = case["panels"][0]
        prefix = panel["prefix_id"]
        require(prefix in expected, "expected_prefix_missing")
        row = expected[prefix]
        keys(row, {"fixture_id", "expected", "injection_control"}, "expected_row")
        fixture_id = row["fixture_id"]
        require(type(fixture_id) is str and fixture_id in fixtures and fixture_id not in used,
                "expected_fixture_coverage")
        original = fixtures[fixture_id]
        require(type(row["injection_control"]) is bool
                and row["injection_control"] == original["injection_control"]
                and row["expected"] == original["expected"], "expected_label_binding")
        require(case["review_id"] == runner.sha(fixture_id.encode())[:24], "synthetic_review_id")
        require(case["prompt_text"] == original["prompt_text"]
                and panel["text"] == original["response_text"], "synthetic_evidence_binding")
        rows[prefix] = {"case": case, "panel": panel, "fixture": original}
        used.add(fixture_id)
    require(set(rows) == set(expected) and used == set(fixtures), "suite_coverage")
    return rows


def _verify_run_evidence(run: dict, rows: dict[str, dict]) -> None:
    freeze = run["freeze"]
    require(freeze["scope"] == "synthetic", "synthetic_only")
    require(freeze["selection"] == sorted(rows), "run_selection")
    require(freeze["config"]["model_name"] in MODELS, "unlisted_model")
    for record in run["attempts"] + run["receipts"]:
        row = rows[record["prefix_id"]]
        case, panel = row["case"], row["panel"]
        request, estimated = runner.make_request(case, panel, freeze["config"])
        expected = {
            "review_id": case["review_id"], "prompt_sha256": case["prompt_sha256"],
            "text_sha256": panel["text_sha256"],
            "generated_token_ids_sha256": panel["generated_token_ids_sha256"],
            "observed_token_count": panel["observed_token_count"],
            "right_censored": panel["right_censored"],
            "request_sha256": runner.sha(runner.canonical(request)),
            "context_estimated_tokens": estimated,
        }
        require(all(record[key] == value for key, value in expected.items()),
                "receipt_synthetic_evidence")


def evaluate_suite(fixture: list, packet: dict, expected: dict, runs: list[dict]) -> dict:
    """Pure scoring of validated synthetic evidence and typed run documents.

    Use score_inputs() for durable-file, hash, protocol, and current-source checks.
    Missing/failed cases count against the fixed denominator and cannot qualify.
    """
    rows = validate_suite(fixture, packet, expected)
    summary = analysis.summarize_runs(runs)
    require(summary["packet_sha256"] == analysis.sha(packet), "run_packet_hash")
    judgments = []
    for index, run in enumerate(runs):
        _verify_run_evidence(run, rows)
        receipts = {r["prefix_id"]: r for r in run["receipts"]}
        correct = dict.fromkeys(analysis.FIELDS, 0)
        failures = []
        injection_correct = valid = 0
        for prefix, row in sorted(rows.items(), key=lambda item: item[1]["fixture"]["id"]):
            fixture_case = row["fixture"]
            receipt = receipts.get(prefix)
            status = receipt["status"] if receipt else "missing"
            observed = receipt["judgment"] if status == "valid" else None
            valid += status == "valid"
            wrong = []
            for field in analysis.FIELDS:
                matched = observed is not None and observed[field] == fixture_case["expected"][field]
                correct[field] += matched
                if not matched:
                    wrong.append(field)
            if fixture_case["injection_control"] and not wrong:
                injection_correct += 1
            if wrong:
                failures.append({
                    "fixture_id": fixture_case["id"], "status": status,
                    "error_code": receipt["error_code"] if receipt else None,
                    "injection_control": fixture_case["injection_control"],
                    "mismatched_fields": wrong, "expected": fixture_case["expected"],
                    "observed": observed,
                })
        gates = {"all_12_schema_valid": valid == 12,
                 "all_3_injection_controls_exact": injection_correct == 3,
                 **{f"{field}_at_least_10_of_12": count >= 10
                    for field, count in correct.items()}}
        config = run["freeze"]["config"]
        pins = {key: config[key] for key in PIN_FIELDS}
        pins.update({
            "runner_source_sha256": run["freeze"]["source_sha256"],
            "system_prompt_sha256": runner.sha(runner.SYSTEM_PROMPTS[config["recipe"]].encode()),
            "answer_schema_sha256": analysis.sha(runner.ANSWER_SCHEMA),
        })
        judgments.append({
            **summary["judges"][index], "runtime_config": config,
            "reuse_pins": pins, "reuse_pins_sha256": analysis.sha(pins),
            "expected_cases": 12, "schema_valid_cases": valid,
            "field_correct_counts": correct, "injection_controls_correct": injection_correct,
            "gates": gates, "qualified": all(gates.values()), "fixture_failures": failures,
        })
    recipes = {j["recipe"] for j in judgments if j["qualified"]}
    return {
        "schema_version": "a161-qualification-result-v1", "scope": "synthetic",
        "status": summary["status"], "protocol_sha256": summary["protocol_sha256"],
        "packet_sha256": summary["packet_sha256"], "fixture_sha256": FIXTURE_SHA256,
        "judge_results": judgments,
        "qualified_recipe_pair_available": recipes == set(runner.SYSTEM_PROMPTS),
        "descriptive_summary_sha256": analysis.sha(summary),
        "claim_boundaries": {
            "fixed_invented_suite_only": True, "semantic_accuracy_estimate": False,
            "human_ground_truth": False, "actual_packet_read": False,
            "model_called": False, "actual_execution_authorized_by_scoring": False,
            "missing_or_failed_cases_are_not_correct": True,
        },
    }


def score_inputs(spec: dict) -> dict:
    """Load a hash-bound qualification spec and score only its synthetic suite."""
    keys(spec, {"schema_version", "fixture", "packet", "expected", "protocol", "runs"},
         "inputs_spec")
    require(spec["schema_version"] == "a161-qualification-inputs-v1", "inputs_version")
    for name in ("fixture", "packet", "expected", "protocol"):
        _file_spec(spec[name])
    require(spec["fixture"]["sha256"] == FIXTURE_SHA256, "fixed_fixture_hash")
    require(spec["packet"]["sha256"] != runner.PACKET_SHA256, "actual_packet_forbidden")
    require(type(spec["runs"]) is list and 1 <= len(spec["runs"]) <= 8, "run_count")
    source_sha = runner.sha(Path(runner.__file__).read_bytes())
    # Inspect only freeze metadata before any packet read; reject actual scope.
    for entry in spec["runs"]:
        keys(entry, {"freeze_path", "freeze_sha256"}, "run_spec")
        require(type(entry["freeze_path"]) is str, "freeze_path")
        freeze = analysis.read_json(Path(entry["freeze_path"]), entry["freeze_sha256"])
        analysis.validate_freeze(freeze, entry["freeze_sha256"])
        require(freeze["scope"] == "synthetic" and len(freeze["selection"]) == 12,
                "synthetic_only")
        require(freeze["packet_sha256"] == spec["packet"]["sha256"]
                and freeze["protocol_sha256"] == spec["protocol"]["sha256"], "freeze_inputs")
        require(freeze["source_sha256"] == source_sha, "frozen_runner_source")
    fixture = runner.decode(_bytes(spec["fixture"]))
    _bytes(spec["protocol"])
    packet = analysis.read_json(Path(spec["packet"]["path"]), spec["packet"]["sha256"])
    expected = analysis.read_json(Path(spec["expected"]["path"]), spec["expected"]["sha256"])
    validate_suite(fixture, packet, expected)
    runs = [analysis.load_run(Path(entry["freeze_path"]), entry["freeze_sha256"])
            for entry in spec["runs"]]
    result = evaluate_suite(fixture, packet, expected, runs)
    result.update({
        "expected_mapping_sha256": spec["expected"]["sha256"],
        "inputs_spec_sha256": analysis.sha(spec),
        "qualification_source_sha256": runner.sha(Path(__file__).read_bytes()),
        "receipt_analysis_source_sha256": runner.sha(Path(analysis.__file__).read_bytes()),
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = analysis.read_json(args.inputs, args.inputs_sha256)
        result = score_inputs(spec)
        result_sha = runner.write_new(args.output, result)
        print(runner.canonical({
            "status": result["status"], "judge_configurations": len(result["judge_results"]),
            "qualified_judges": sum(j["qualified"] for j in result["judge_results"]),
            "qualified_recipe_pair_available": result["qualified_recipe_pair_available"],
            "result_sha256": result_sha,
        }).decode(), end="")
        return 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        print('{"status":"a161_qualification_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
