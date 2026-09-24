"""A191 fresh paired answer-availability tasks and fixed-cohort analysis.

Pure standard library only. Native generation provenance and model/runtime
qualification belong to the coordinator, not to these input and scoring rules.
"""

from __future__ import annotations

import hashlib
import json
import re

NAMESPACE = "a191-answer-availability-v1"
SCHEMA = "a191-answer-availability-tasks-v1"
CORE_COUNT = 8
CONDITIONS = ("computed", "supplied")
ROW_COUNT = 16
ITEM_COUNT = 2
CAP = 64
SYSTEM_INSTRUCTION = (
    "Each of the two data rows contains a left integer, a right integer, and an answer field. "
    "If the answer field is ?, add the two integers; otherwise copy the supplied answer. "
    "Preserve row order. Output exactly two comma-separated base-10 integer answers, "
    "with no other text."
)
ASCII_WHITESPACE = " \t\r\n\v\f"
SCORE_KEYS = ("outcome_status", "label", "terminal_eos", "capped", "censored")
IDENTITY_KEYS = (
    "observation_id",
    "core_index",
    "core_id",
    "group_id",
    "condition",
    "sequence_index",
)
PAIR_CATEGORIES = ("both_success", "both_failure", "supplied_only", "computed_only", "unknown")
_INTEGER = re.compile(r"[+-]?[0-9]+", flags=re.ASCII)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def operand(core_index, item_index, side):
    _require(type(core_index) is int and 0 <= core_index < CORE_COUNT, "core_index")
    _require(type(item_index) is int and 0 <= item_index < ITEM_COUNT, "item_index")
    _require(type(side) is int and side in (0, 1), "operand_side")
    raw = hashlib.sha256(f"{NAMESPACE}|{core_index}|{item_index}|{side}".encode("ascii")).digest()
    absolute = 10 + int.from_bytes(raw, "big") % 90
    return -absolute if raw[0] & 1 else absolute


def core_operands(core_index):
    return [[operand(core_index, item, side) for side in (0, 1)] for item in range(ITEM_COUNT)]


def _strata(pairs):
    counts = {
        name: 0
        for name in (
            "positive_positive_items",
            "negative_negative_items",
            "opposite_sign_items",
            "units_carry_items",
            "units_borrow_items",
            "zero_sum_items",
        )
    }
    for a, b in pairs:
        sign = (
            "positive_positive"
            if a > 0 and b > 0
            else "negative_negative"
            if a < 0 and b < 0
            else "opposite_sign"
        )
        counts[sign + "_items"] += 1
        counts["units_carry_items"] += int(a * b > 0 and abs(a) % 10 + abs(b) % 10 >= 10)
        counts["units_borrow_items"] += int(
            a * b < 0 and max(abs(a), abs(b)) % 10 < min(abs(a), abs(b)) % 10
        )
        counts["zero_sum_items"] += int(a + b == 0)
    return {key: str(value) for key, value in counts.items()}


def build_roster():
    """Hash-order eight cores; alternate which condition is first within each pair."""
    core_order = sorted(
        (hashlib.sha256(f"{NAMESPACE}|schedule|{core}".encode("ascii")).hexdigest(), core)
        for core in range(CORE_COUNT)
    )
    rows = []
    for pair_index, (schedule_sha, core) in enumerate(core_order):
        order = CONDITIONS if pair_index % 2 == 0 else tuple(reversed(CONDITIONS))
        for within_pair_index, condition in enumerate(order):
            pairs = core_operands(core)
            answers = [a + b for a, b in pairs]
            user = "\n".join(
                f"{a} {b} {answer if condition == 'supplied' else '?'}"
                for (a, b), answer in zip(pairs, answers, strict=True)
            )
            group = f"core_{core:02d}"
            rows.append(
                {
                    "observation_id": f"{group}_{condition}",
                    "core_index": core,
                    "core_id": group,
                    "group_id": group,
                    "condition": condition,
                    "operands": pairs,
                    "expected_sums": answers,
                    "strata": _strata(pairs),
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": user},
                    ],
                    "core_schedule_sha256": schedule_sha,
                    "pair_index": pair_index,
                    "within_pair_index": within_pair_index,
                    "sequence_index": len(rows),
                }
            )
    return rows


compile_roster = build_roster


def validate_roster(rows):
    _require(type(rows) is list, "roster_type")
    expected = build_roster()
    _require(_canonical(rows) == _canonical(expected), "roster_mismatch")
    return expected


def validate_row(row):
    _require(type(row) is dict, "row_type")
    _require(
        type(row.get("core_index")) is int and 0 <= row["core_index"] < CORE_COUNT, "core_index"
    )
    _require(type(row.get("condition")) is str and row["condition"] in CONDITIONS, "condition")
    expected = next(
        r
        for r in build_roster()
        if r["core_index"] == row["core_index"] and r["condition"] == row["condition"]
    )
    _require(_canonical(row) == _canonical(expected), "row_mismatch")
    return expected


def _normalized_integer(field):
    field = field.strip(ASCII_WHITESPACE)
    if _INTEGER.fullmatch(field) is None:
        return None
    negative = field[0] == "-"
    digits = field[1:] if field[0] in "+-" else field
    digits = digits.lstrip("0") or "0"
    return "-" + digits if negative and digits != "0" else digits


def oracle_response(text, expected_sums):
    _require(type(text) is str, "response_text")
    _require(
        type(expected_sums) is list
        and len(expected_sums) == ITEM_COUNT
        and all(type(x) is int for x in expected_sums),
        "expected_sums",
    )
    fields = text.split(",")
    return len(fields) == ITEM_COUNT and [_normalized_integer(x) for x in fields] == [
        str(x) for x in expected_sums
    ]


def outcome_for_generation(row, generation, decoded_text, eos_token_ids):
    """EOS and exact complete content are required; incomplete parents stay unknown."""
    row = validate_row(row)
    _require(type(generation) is dict, "generation_type")
    _require({"status", "token_ids", "stop_reason"} <= generation.keys(), "generation_fields")
    _require(
        type(eos_token_ids) is list
        and bool(eos_token_ids)
        and all(type(x) is int and x >= 0 for x in eos_token_ids)
        and len(eos_token_ids) == len(set(eos_token_ids)),
        "eos_token_ids",
    )
    tokens = generation.get("token_ids")
    _require(
        type(tokens) is list
        and len(tokens) <= CAP
        and all(type(x) is int and x >= 0 for x in tokens),
        "generation_tokens",
    )
    if "observed_tokens" in generation:
        _require(
            type(generation["observed_tokens"]) is int
            and generation["observed_tokens"] == len(tokens),
            "observed_tokens",
        )
    _require(not any(x in eos_token_ids for x in tokens[:-1]), "nonterminal_eos")
    terminal = bool(tokens and tokens[-1] in eos_token_ids)
    status, stop = generation.get("status"), generation.get("stop_reason")
    _require(
        type(status) is str and status in ("completed", "infrastructure_failed", "unattempted"),
        "generation_status",
    )
    if status == "completed":
        _require(
            (stop == "eos" and terminal) or (stop == "cap" and not terminal and len(tokens) == CAP),
            "generation_stop",
        )
        _require(type(decoded_text) is str, "decoded_text")
        label = int(terminal and oracle_response(decoded_text, row["expected_sums"]))
    else:
        _require(stop is None, "generation_stop")
        _require(status != "unattempted" or not tokens, "unattempted_tokens")
        _require(decoded_text is None or type(decoded_text) is str, "decoded_text")
        _require(status != "unattempted" or decoded_text is None, "unattempted_text")
        label = None
    return {
        "status": "known" if label is not None else "missing",
        "label": label,
        "reason": None if label is not None else "not_measured",
        "terminal_eos": None if status == "unattempted" else terminal,
        "capped": None if status == "unattempted" else stop == "cap",
        "censored": None if status == "unattempted" else not terminal and len(tokens) < CAP,
    }


def score_generation(row, generation, decoded_text, eos_token_ids):
    outcome = outcome_for_generation(row, generation, decoded_text, eos_token_ids)
    return {
        "outcome_status": outcome["status"],
        **{key: outcome[key] for key in SCORE_KEYS if key != "outcome_status"},
    }


def _validate_record(row, record):
    _require(type(record) is dict, "record_type")
    for key in IDENTITY_KEYS:
        _require(
            key in record and _canonical(record[key]) == _canonical(row[key]), "record_identity"
        )
    _require(all(key in record for key in SCORE_KEYS), "record_score_keys")
    label = record["label"]
    _require(label is None or type(label) is int and label in (0, 1), "record_label")
    _require(
        record["outcome_status"] == ("missing" if label is None else "known"), "record_outcome"
    )
    flags = [record[key] for key in ("terminal_eos", "capped", "censored")]
    unobserved = all(x is None for x in flags)
    _require(unobserved or all(type(x) is bool for x in flags), "record_stop_types")
    _require(not unobserved or label is None, "record_unobserved_label")
    _require(
        not (record["terminal_eos"] and (record["capped"] or record["censored"])), "record_stop"
    )
    _require(not (record["capped"] and record["censored"]), "record_stop")
    _require(label != 1 or record["terminal_eos"] is True, "record_success")
    _require(label is None or record["terminal_eos"] or record["capped"], "record_completion")


def _summary(records):
    labels = [r["label"] for r in records]
    planned, known = len(labels), sum(x is not None for x in labels)
    success = sum(x == 1 for x in labels)
    return {
        "planned_rows": planned,
        "success_rate": {
            "point": success / planned if known == planned else None,
            "lower": success / planned,
            "upper": (success + planned - known) / planned,
            "planned_outcomes": planned,
            "resolved_outcomes": known,
        },
        "labels": {
            "success": success,
            "failure": sum(x == 0 for x in labels),
            "unknown": planned - known,
        },
        "terminal_eos": sum(r["terminal_eos"] is True for r in records),
        "terminal_eos_unknown": sum(r["terminal_eos"] is None for r in records),
        "capped": sum(r["capped"] is True for r in records),
        "capped_unknown": sum(r["capped"] is None for r in records),
        "censored": {
            "true": sum(r["censored"] is True for r in records),
            "false": sum(r["censored"] is False for r in records),
            "unknown": sum(r["censored"] is None for r in records),
        },
    }


def aggregate_records(rows, records):
    """Retain all16 slots and all8 pairs; never return an observed-subset point."""
    rows = validate_roster(rows)
    _require(type(records) is list and len(records) == ROW_COUNT, "record_slots")
    pairs = {}
    for row, record in zip(rows, records, strict=True):
        _validate_record(row, record)
        pairs.setdefault(row["core_index"], {})[row["condition"]] = record["label"]
    _require(
        len(pairs) == CORE_COUNT and all(set(pair) == set(CONDITIONS) for pair in pairs.values()),
        "paired_cohort",
    )
    lower = upper = known = resolved_pairs = 0
    categories = dict.fromkeys(PAIR_CATEGORIES, 0)
    for pair in pairs.values():
        supplied, computed = pair["supplied"], pair["computed"]
        lower += (0 if supplied is None else supplied) - (1 if computed is None else computed)
        upper += (1 if supplied is None else supplied) - (0 if computed is None else computed)
        known += int(supplied is not None) + int(computed is not None)
        if supplied is None or computed is None:
            categories["unknown"] += 1
        else:
            resolved_pairs += 1
            category = {
                (1, 1): "both_success",
                (0, 0): "both_failure",
                (1, 0): "supplied_only",
                (0, 1): "computed_only",
            }[(supplied, computed)]
            categories[category] += 1
    contrast = {
        "point": lower / CORE_COUNT if known == ROW_COUNT else None,
        "lower": lower / CORE_COUNT,
        "upper": upper / CORE_COUNT,
        "planned_outcomes": ROW_COUNT,
        "resolved_outcomes": known,
        "planned_pairs": CORE_COUNT,
        "resolved_pairs": resolved_pairs,
    }
    supplied_labels = [r["label"] for r in records if r["condition"] == "supplied"]
    supplied_qualification = (
        False if 0 in supplied_labels else True if all(x == 1 for x in supplied_labels) else None
    )
    return {
        "schema_version": "a191-answer-availability-analysis-v1",
        "overall": _summary(records),
        "by_condition": {
            condition: _summary([r for r in records if r["condition"] == condition])
            for condition in CONDITIONS
        },
        "contrasts": {"success_rate": {"supplied_minus_computed": contrast}},
        "supplied_qualification": supplied_qualification,
        "paired_outcomes": {
            "planned_pairs": CORE_COUNT,
            "resolved_pairs": resolved_pairs,
            "counts": categories,
        },
    }


analyze = aggregate_records
