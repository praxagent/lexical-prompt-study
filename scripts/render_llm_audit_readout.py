#!/usr/bin/env python3
"""Render only an A162 numeric aggregate; never open packets, receipts, or APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from lexical_prompt_study.llm_audit_analysis import FIELDS, HORIZONS, read_json


FIELD_NAMES = {"disallowed_assistance": "Disallowed assistance", "safe_task_completion": "Safe-task completion",
               "refusal_present": "Refusal present"}
RECIPE_NAMES = {"assistance_first": "Assistance first", "utility_first": "Utility first"}
STAGES = {"development": "Development (synthetic)", "holdout": "Holdout (synthetic)", "actual": "Selected study sample"}
STATUS_NAMES = {"not_started": "Not started", "incomplete": "Incomplete", "complete_valid": "Complete; all outputs valid",
                "complete_with_failures": "Complete; some outputs failed"}


def require(ok):
    if not ok:
        raise ValueError("invalid_numeric_aggregate")


def count(value):
    require(type(value) is int and 0 <= value <= 10_000_000)
    return value


def fraction(numerator, denominator):
    numerator, denominator = count(numerator), count(denominator)
    require(numerator <= denominator)
    return f"{numerator}/{denominator}" + (f" ({100 * numerator / denominator:.1f}%)" if denominator else " (unavailable)")


def table(headings, rows):
    return ["| " + " | ".join(headings) + " |", "| " + " | ".join("---" for _ in headings) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows), ""]


def coverage(row, expected):
    require(count(row["expected"]) == expected)
    require(sum(count(row[key]) for key in ("valid", "missing", "schema_failed", "runtime_failed", "context_rejected")) == expected)
    for field, values in FIELDS.items():
        require(set(row["judgments"][field]) == set(values))
        require(sum(count(row["judgments"][field][value]) for value in values) == row["valid"])


def render(summary: dict, *, aggregate_sha256: str) -> str:
    """A deterministic readout of validated aggregate fields, with no source-file reads."""
    require(summary["schema_version"] == "a162-audit-summary-v1" and summary["instrument_id"] == "a162"
            and summary["provider"] == "openai" and summary["judge_kind"] == "llm")
    stage = summary["stage"]
    require(stage in STAGES and summary["status"] in STATUS_NAMES)
    require(summary["scope"] == ("verified_a156" if stage == "actual" else "synthetic"))
    require(type(aggregate_sha256) is str and len(aggregate_sha256) == 64
            and all(c in "0123456789abcdef" for c in aggregate_sha256))
    overall, judges = summary["overall"], summary["judges"]
    prefixes = count(overall["unique_prefixes"])
    require(prefixes == {"development": 12, "holdout": 24, "actual": 287}[stage])
    require(type(judges) is list and 1 <= len(judges) <= 8)
    require(len(overall["coverage_by_judge"]) == len(judges))
    for index, judge in enumerate(judges):
        require(judge["judge_index"] == index and judge["provider"] == "openai"
                and judge["model_name"] == "gpt-4.1-2025-04-14" and judge["recipe"] in RECIPE_NAMES)
        coverage(overall["coverage_by_judge"][index], prefixes)
    descriptions = overall["mapped_descriptions"]
    mapped = descriptions["status"] == "available"
    lines = ["# A162 model-judge audit readout", "", f"**{STAGES[stage]}. {STATUS_NAMES[summary['status']]}.**", ""]
    if mapped:
        cases, bindings = count(descriptions["selected_cases"]), count(descriptions["horizon_bindings"])
        aliases = count(descriptions["repeated_prefix_aliases"])
        require(bindings == 4 * cases and aliases == bindings - prefixes)
        if stage == "actual":
            require((cases, prefixes, bindings) == (87, 287, 348))
        lines += [f"{cases} selected cases contain {prefixes} distinct response prefixes and {bindings} horizon bindings "
                  f"at 128, 256, 512 and 1,024 tokens. The {aliases} additional bindings reuse prefixes; "
                  "they are not additional model judgments.", ""]
    else:
        lines += [f"{prefixes} distinct prefixes. Case, horizon and classifier descriptions are unavailable because "
                  "the aggregate has no numeric mapping.", ""]
    lines += ["These are model judgments on a selected sample, not human ratings or ground truth. "
              "Agreement measures concordance, not accuracy. Shared models, recipes, request cores and repeated "
              "prefixes are not independent observations. Missing or failed outputs are not negative judgments.", "",
              "## Judge coverage", "", "All recipes use OpenAI `gpt-4.1-2025-04-14`.", ""]
    rows = []
    for index, judge in enumerate(judges):
        row = overall["coverage_by_judge"][index]
        rows.append([f"J{index + 1}", RECIPE_NAMES[judge["recipe"]], fraction(row["valid"], prefixes),
                     count(row["missing"]), count(row["schema_failed"]), count(row["runtime_failed"]),
                     count(row["context_rejected"])])
    lines += table(["Judge", "Recipe", "Valid / selected", "Missing", "Schema failed", "Runtime failed", "Context rejected"], rows)
    rows = []
    for index, row in enumerate(overall["coverage_by_judge"]):
        for field, values in FIELDS.items():
            counts = "; ".join(f"{value.replace('_', ' ')}: {count(row['judgments'][field][value])}" for value in values)
            rows.append([f"J{index + 1}", FIELD_NAMES[field], count(row["valid"]), counts])
    lines += table(["Judge", "Field", "Valid denominator", "Label counts"], rows)
    lines += ["## Agreement between recipes", ""]
    pairs = overall["paired_concordance"]
    if not pairs:
        lines += ["Unavailable: only one judge configuration is represented.", ""]
    else:
        rows = []
        for pair in pairs:
            left, right = pair["judge_indices"]
            require(type(left) is int and type(right) is int and 0 <= left < right < len(judges))
            counts = pair["counts"]
            require(count(counts["expected_prefix_pairs"]) == prefixes)
            valid = count(counts["both_valid"])
            require(valid + count(counts["excluded_failed_or_missing"]) == prefixes)
            for field in FIELDS:
                row = counts["fields"][field]
                require(count(row["agree"]) + count(row["disagree"]) == valid)
                rows.append([f"J{left + 1} / J{right + 1}", FIELD_NAMES[field], fraction(valid, prefixes),
                             fraction(row["agree"], valid), count(row["disagree"]), count(row["uncertainty_in_pair"]),
                             fraction(row["definite_agree"], row["both_definite"])])
        lines += table(["Pair", "Field", "Both valid / selected", "Agree / both valid", "Disagree", "Either uncertain",
                        "Agree / neither uncertain"], rows)
        lines += ["Uncertain pairs can also agree or disagree, so that column overlaps the others. "
                  "“Neither uncertain” retains the safe-task category “not applicable”.", ""]
    if len(judges) > 2 and overall["all_judges_concordance"]["available"]:
        unanimity = overall["all_judges_concordance"]
        all_valid = count(unanimity["all_judges_valid_prefixes"])
        lines += [f"All configurations are valid on {fraction(all_valid, prefixes)} prefixes.", ""]
        rows = []
        for field, values in FIELDS.items():
            row = unanimity["fields"][field]
            require(sum(count(row["all_same"][value]) for value in values) + count(row["disagree"]) == all_valid)
            rows.append([FIELD_NAMES[field], "; ".join(f"{v.replace('_', ' ')}: {count(row['all_same'][v])}" for v in values),
                         count(row["disagree"]), count(row["any_uncertain"])])
        lines += table(["Field", "Unanimous label counts", "Disagreement", "Any uncertain"], rows)
    lines += ["## Classifier concordance by horizon", ""]
    if not mapped:
        lines += ["Unavailable: no source-bound numeric classifier mapping was supplied.", ""]
    else:
        rows = []
        sources = descriptions["classifier_concordance_by_judge_and_horizon"]
        require(len(sources) == len(judges))
        for index, horizons in enumerate(sources):
            for horizon in HORIZONS:
                row = horizons[str(horizon)]
                require(count(row["expected"]) == cases)
                require(count(row["valid"]) + count(row["excluded_failed_or_missing"]) == cases)
                require(count(row["agree"]) + count(row["disagree"]) == count(row["binary_comparable"]))
                rows.append([f"J{index + 1}", horizon, cases, fraction(row["agree"], row["binary_comparable"]),
                             count(row["table"]["negative"]["yes"]), count(row["table"]["positive"]["no"]),
                             count(row["judge_uncertain"]), count(row["classifier_unknown"]),
                             count(row["excluded_failed_or_missing"])])
        lines += table(["Judge", "Horizon", "Selected", "Agree / binary comparable", "Classifier − / judge yes",
                        "Classifier + / judge no", "Judge uncertain", "Classifier unknown", "Failed / missing"], rows)
        lines += ["Binary comparisons exclude unknown classifier labels and uncertain judge answers. "
                  "Those exclusions can overlap. Directional differences are not false-positive or false-negative estimates.", ""]
    lines += ["## Semantic and mechanical utility", ""]
    utility = summary.get("semantic_vs_mechanical_utility")
    if utility is None:
        lines += ["Unavailable: no separately bound mechanical utility input was supplied. "
                  "Safety or refusal labels are not substitutes for utility.", ""]
    else:
        require(mapped and utility["schema_version"] == "a161-utility-comparison-v1"
                and utility["mechanical_ground_truth"] is False and len(utility["per_judge"]) == len(judges))
        rows, refusal_rows, horizon_rows = [], [], []
        for index, groups in enumerate(utility["per_judge"]):
            group = groups["distinct_prefixes"]
            require(count(group["expected"]) == prefixes and count(group["valid"]) + count(group["failed_or_missing"]) == prefixes)
            for label, name in (("exact", "Exact match"), ("not_exact", "No exact match"), ("not_applicable", "Not applicable")):
                row = group["safe_task_table"][label]
                values = [count(row[value]) for value in FIELDS["safe_task_completion"]]
                rows.append([f"J{index + 1}", name, sum(values), *values])
            for parser, name in (("detected", "Detected"), ("not_detected", "Not detected")):
                row = group["refusal_table"][parser]
                values = [count(row[value]) for value in FIELDS["refusal_present"]]
                refusal_rows.append([f"J{index + 1}", name, sum(values), *values])
            for view in ("distinct_prefixes", *(str(h) for h in HORIZONS)):
                row = groups[view]
                exact, not_exact = row["safe_task_table"]["exact"], row["safe_task_table"]["not_exact"]
                exact_total = sum(count(exact[v]) for v in FIELDS["safe_task_completion"])
                safe_total = exact_total + sum(count(not_exact[v]) for v in FIELDS["safe_task_completion"])
                semantic_complete = count(exact["complete"]) + count(not_exact["complete"])
                uncertain_na = sum(count(table_[v]) for table_ in (exact, not_exact) for v in ("uncertain", "not_applicable"))
                horizon_rows.append([f"J{index + 1}", "Distinct prefixes" if view == "distinct_prefixes" else view,
                                     safe_total, fraction(exact_total, safe_total), fraction(semantic_complete, safe_total),
                                     count(exact["complete"]), count(not_exact["complete"]), uncertain_na,
                                     count(row["failed_or_missing"])])
        lines += ["Distinct-prefix cross-tabulations below include valid model outputs only. "
                  "“Not applicable” in the mechanical column denotes direct unsafe requests with no safe-task exact-match score.", ""]
        lines += table(["Judge", "Mechanical result", "Valid count", "LLM complete", "LLM incomplete", "LLM uncertain", "LLM not applicable"], rows)
        lines += ["Safe-task views exclude mechanically inapplicable requests. Failed/missing counts cover the full selected view.", ""]
        lines += table(["Judge", "View", "Valid safe-task count", "Mechanical exact", "LLM complete", "Both", "LLM complete / not exact",
                        "LLM uncertain or N/A", "All failed / missing"], horizon_rows)
        lines += ["Refusal parser comparison, using distinct prefixes:", ""]
        lines += table(["Judge", "Mechanical refusal parser", "Valid count", "LLM yes", "LLM no", "LLM uncertain"], refusal_rows)
        lines += ["Exact matching and refusal parsing are mechanical measures, not semantic truth.", ""]
    lines += ["## Interpretation and costs", "", "The sample was selected across placements, direct/safe frames and classifier-trajectory strata. "
              "It is not a random sample of ordinary requests; these counts do not estimate population prevalence. "
              "This audit does not establish first harmful-token timing, prevention, or a deployable defense.", ""]
    rows = []
    for index, judge in enumerate(judges):
        bill = judge.get("billing")
        if bill is not None:
            amounts = [count(bill[key]) for key in ("metered_micro_usd", "reserved_unknown_micro_usd", "interrupted_reserved_micro_usd")]
            rows.append([f"J{index + 1}", *(f"${value / 1_000_000:.6f}" for value in amounts)])
    if rows:
        lines += table(["Judge", "Metered token-price estimate", "Unknown-usage reserve", "Interrupted-attempt reserve"], rows)
        lines += ["These amounts cover the summarized runs only. They exclude other qualification/development runs, "
                  "are not settled invoices, and do not independently verify the shared budget ledger.", ""]
    else:
        lines += ["Fee estimates are unavailable in this aggregate.", ""]
    lines += [f"Aggregate SHA256: `{aggregate_sha256}`.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--aggregate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = read_json(args.aggregate, args.aggregate_sha256)
        result = render(summary, aggregate_sha256=args.aggregate_sha256).encode("utf-8")
        output = args.output.absolute()
        require(not any(path.is_symlink() for path in (output, *output.parents)))
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(result)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"status": "rendered", "report_sha256": hashlib.sha256(result).hexdigest()}, sort_keys=True))
        return 0
    except Exception:
        print('{"status":"readout_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
